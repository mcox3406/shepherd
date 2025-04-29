#!/usr/bin/env python3
"""
Analyze and compare inference scaling experiments from ShEPhERD.

This script helps analyze and visualize results from multiple inference scaling
experiments to enable effective comparison of different search algorithms 
and hyperparameter settings.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw

from shepherd.inference_scaling import create_rdkit_molecule


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze inference scaling experiments')
    parser.add_argument('--base_dir', type=str, default='inference_scaling_experiments',
                        help='Base directory containing experiment results')
    parser.add_argument('--experiments', type=str, nargs='*', default=None,
                        help='List of specific experiment names to analyze (default: all)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results (default: base_dir/analysis)')
    parser.add_argument('--draw_molecules', action='store_true',
                        help='Generate images of the best molecules')
    return parser.parse_args()


def load_experiment_results(experiment_path):
    """Load experiment results from a directory"""
    results_path = experiment_path / "results.pkl"
    summary_path = experiment_path / "summary.json"
    
    if not results_path.exists():
        print(f"Warning: No results.pkl found in {experiment_path}")
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return json.load(f)
        return None
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


def extract_summary_data(results):
    """Extract key summary data from experiment results"""
    if results is None:
        return None
    
    summary = {
        'algorithm': results.get('algorithm', 'unknown'),
        'best_score': results.get('best_score', 0),
        'baseline_score': results.get('baseline_score', 0),
        'improvement': results.get('best_score', 0) - results.get('baseline_score', 0),
        'improvement_pct': (results.get('best_score', 0) - results.get('baseline_score', 0)) / max(0.0001, results.get('baseline_score', 0)) * 100,
        'duration_seconds': results.get('duration_seconds', 0),
        'num_evaluations': len(results.get('scores', [])),
    }
    
    # add configuration details
    config = results.get('config', {})
    if isinstance(config, dict):
        if 'algorithm' in config:
            summary['algorithm'] = config['algorithm']
        if config.get('algorithm') == 'random':
            summary['num_trials'] = config.get('num_trials', 0)
        elif config.get('algorithm') == 'zero_order':
            summary['num_steps'] = config.get('num_steps', 0)
            summary['num_neighbors'] = config.get('num_neighbors', 0)
            summary['step_size'] = config.get('step_size', 0)
    
    # extract SMILES if available
    if 'best_smiles' in results:
        summary['best_smiles'] = results['best_smiles']
    elif 'best_sample' in results:
        try:
            mol = create_rdkit_molecule(results['best_sample'])
            if mol is not None:
                summary['best_smiles'] = Chem.MolToSmiles(mol)
        except:
            pass
    
    return summary


def create_summary_table(summaries):
    """Create a summary table from experiment summaries"""
    if not summaries:
        return None
    
    df = pd.DataFrame(summaries)
    df.index = df['experiment_name']
    
    # determine which columns to display based on what's available
    display_columns = ['algorithm', 'best_score', 'baseline_score', 'improvement', 
                       'improvement_pct', 'duration_seconds', 'num_evaluations']
    
    # add algorithm-specific columns
    if 'random' in df['algorithm'].values:
        if 'num_trials' in df.columns:
            display_columns.append('num_trials')
    
    if 'zero_order' in df['algorithm'].values:
        for col in ['num_steps', 'num_neighbors', 'step_size']:
            if col in df.columns:
                display_columns.append(col)
    
    # create formatted table
    table = df[display_columns].copy()
    
    # format numeric columns
    if 'best_score' in table.columns:
        table['best_score'] = table['best_score'].map('{:.4f}'.format)
    if 'baseline_score' in table.columns:
        table['baseline_score'] = table['baseline_score'].map('{:.4f}'.format)
    if 'improvement' in table.columns:
        table['improvement'] = table['improvement'].map('{:.4f}'.format)
    if 'improvement_pct' in table.columns:
        table['improvement_pct'] = table['improvement_pct'].map('{:.2f}%'.format)
    if 'duration_seconds' in table.columns:
        # format duration as minutes if > 60 seconds
        table['duration'] = df['duration_seconds'].apply(
            lambda x: f"{x:.1f}s" if x < 60 else f"{x/60:.1f}m"
        )
        # remove original duration_seconds column
        table = table.drop('duration_seconds', axis=1)
        display_columns[display_columns.index('duration_seconds')] = 'duration'
    
    # rename columns for display
    rename_map = {
        'num_evaluations': 'evals',
        'num_trials': 'trials',
        'num_steps': 'steps',
        'num_neighbors': 'neighbors',
        'step_size': 'step_size',
        'improvement_pct': 'improv_%'
    }
    table = table.rename(columns=rename_map)
    
    return table


def plot_score_comparison(experiments, results_dict, output_dir):
    """Create plots comparing score progression across experiments"""
    plt.figure(figsize=(12, 8))
    
    # plot score progression for each experiment
    for exp_name in experiments:
        results = results_dict.get(exp_name)
        if results is None or 'scores' not in results:
            continue
        
        scores = results['scores']
        
        # get algorithm and parameters for the label
        algorithm = results.get('algorithm', 'unknown')
        config = results.get('config', {})
        
        if algorithm == 'random':
            label = f"Random (trials={config.get('num_trials', '?')})"
        elif algorithm == 'zero_order':
            label = f"ZO (steps={config.get('num_steps', '?')}, nbrs={config.get('num_neighbors', '?')}, ss={config.get('step_size', '?')})"
        else:
            label = exp_name
        
        # plot scores
        plt.plot(range(1, len(scores) + 1), scores, '-o', label=label, alpha=0.7, markersize=4)
        
        # highlight best score
        best_idx = np.argmax(scores)
        plt.scatter(best_idx + 1, scores[best_idx], marker='*', s=100)
    
    # add baseline if available (use the first experiment's baseline)
    for exp_name in experiments:
        results = results_dict.get(exp_name)
        if results and 'baseline_score' in results:
            plt.axhline(y=results['baseline_score'], color='r', linestyle='--',
                        label=f"Baseline ({results['baseline_score']:.4f})")
            break
    
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Score')
    plt.title('Comparison of Search Algorithms')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_comparison.png")
    print(f"Score comparison plot saved to {output_dir}/score_comparison.png")
    
    # create a plot highlighting improvement over baseline
    plt.figure(figsize=(10, 6))
    improvements = []
    labels = []
    
    for exp_name in experiments:
        results = results_dict.get(exp_name)
        if results is None or 'best_score' not in results or 'baseline_score' not in results:
            continue
        
        improvement = results['best_score'] - results['baseline_score']
        improvements.append(improvement)
        
        # create label
        algorithm = results.get('algorithm', 'unknown')
        if algorithm == 'random':
            label = f"Random\n(t={results.get('config', {}).get('num_trials', '?')})"
        elif algorithm == 'zero_order':
            label = f"ZO\n(s={results.get('config', {}).get('num_steps', '?')}, n={results.get('config', {}).get('num_neighbors', '?')})"
        else:
            label = exp_name.split('_')[0]
        
        labels.append(label)
    
    # plot bar chart
    plt.bar(labels, improvements)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Score Improvement over Baseline')
    plt.title('Comparison of Search Algorithm Improvements')
    plt.grid(axis='y', alpha=0.3)
    
    # add value labels on top of bars
    for i, v in enumerate(improvements):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_comparison.png")
    print(f"Improvement comparison plot saved to {output_dir}/improvement_comparison.png")


def draw_best_molecules(summaries, output_dir):
    """Draw and save images of the best molecules from each experiment"""
    mols = []
    labels = []
    
    for summary in summaries:
        if 'best_smiles' in summary:
            try:
                mol = Chem.MolFromSmiles(summary['best_smiles'])
                if mol is not None:
                    mols.append(mol)
                    
                    # create label with experiment name and score
                    label = f"{summary['experiment_name']}\nScore: {summary['best_score']:.4f}"
                    labels.append(label)
            except:
                pass
    
    if not mols:
        print("No valid molecules to draw")
        return
    
    # draw molecules in a grid
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300), legends=labels)
    img.save(output_dir / "best_molecules.png")
    print(f"Best molecules image saved to {output_dir}/best_molecules.png")


def analyze_experiments(args):
    """Analyze inference scaling experiments"""
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # set up output directory
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # find experiment directories
    if args.experiments:
        # use specified experiment names
        experiment_dirs = [base_dir / exp for exp in args.experiments if (base_dir / exp).exists()]
    else:
        # find all directories with results.pkl or summary.json
        experiment_dirs = []
        for path in base_dir.iterdir():
            if path.is_dir() and ((path / "results.pkl").exists() or (path / "summary.json").exists()):
                experiment_dirs.append(path)
    
    if not experiment_dirs:
        print(f"No experiment results found in {base_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment(s) to analyze")
    
    # load experiment results
    results_dict = {}
    summaries = []
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"Loading experiment: {exp_name}")
        
        results = load_experiment_results(exp_dir)
        if results:
            results_dict[exp_name] = results
            
            # extract summary data
            summary = extract_summary_data(results)
            if summary:
                summary['experiment_name'] = exp_name
                summary['path'] = str(exp_dir)
                summaries.append(summary)
    
    if not summaries:
        print("No valid experiment results found")
        return
    
    # create summary table
    summary_table = create_summary_table(summaries)
    if summary_table is not None:
        print("\nExperiment Summary:")
        print(summary_table)
        
        # save to CSV
        summary_table.to_csv(output_dir / "experiment_summary.csv")
        print(f"Summary table saved to {output_dir}/experiment_summary.csv")
    
    # plot score comparison
    plot_score_comparison(list(results_dict.keys()), results_dict, output_dir)
    
    # draw best molecules if requested
    if args.draw_molecules:
        try:
            draw_best_molecules(summaries, output_dir)
        except Exception as e:
            print(f"Error drawing molecules: {e}")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    analyze_experiments(args) 