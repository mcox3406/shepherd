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
import logging

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
        'nfe': results.get('nfe', 0),
        'best_sa_score': results.get('best_sa_score'),
        'best_clogp_score': results.get('best_clogp_score'),
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
    
    # add NFE and best individual scores if available
    for col in ['nfe', 'best_sa_score', 'best_clogp_score']:
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
    if 'best_sa_score' in table.columns:
        table['best_sa_score'] = table['best_sa_score'].map(lambda x: '{:.4f}'.format(x) if pd.notna(x) else 'N/A')
    if 'best_clogp_score' in table.columns:
        table['best_clogp_score'] = table['best_clogp_score'].map(lambda x: '{:.4f}'.format(x) if pd.notna(x) else 'N/A')
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
        'improvement_pct': 'improv_%',
        'nfe': 'NFE',
        'best_sa_score': 'Best SA',
        'best_clogp_score': 'Best cLogP'
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
    plt.figure(figsize=(12, 6))
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
    
    # plot bar chart with wider bars and more space
    x_pos = np.arange(len(labels))
    bar_width = 0.6
    plt.figure(figsize=(max(12, len(labels)*1.5), 6))
    
    bars = plt.bar(x_pos, improvements, width=bar_width)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Score Improvement over Baseline')
    plt.title('Comparison of Search Algorithm Improvements')
    plt.grid(axis='y', alpha=0.3)
    
    plt.xticks(x_pos, labels)
    if len(labels) > 6:
        plt.xticks(rotation=45, ha='right')
    
    # add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.001,  # Small offset above bar
            f"{improvements[i]:.4f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_comparison.png")
    print(f"Improvement comparison plot saved to {output_dir}/improvement_comparison.png")


def draw_best_molecules(summaries, output_dir):
    """Draw and save images of the best molecules from each experiment"""
    mols = []
    labels = []
    logging.info("Attempting to draw best molecules...")

    for summary in summaries:
        exp_name = summary.get('experiment_name', 'UnknownExp')
        if 'best_smiles' in summary and summary['best_smiles'] and not isinstance(summary['best_smiles'], (int, float)) :
            smiles_str = summary['best_smiles']
            logging.info(f"Found SMILES '{smiles_str}' for {exp_name}")
            try:
                mol = Chem.MolFromSmiles(smiles_str)
                if mol is not None:
                    logging.info(f"Successfully created RDKit Mol for {exp_name}")
                    mols.append(mol)
                    # create label with experiment name and score
                    label = f"{exp_name}\nScore: {summary['best_score']:.4f}"
                    labels.append(label)
                else:
                    logging.warning(f"RDKit MolFromSmiles returned None for SMILES '{smiles_str}' from {exp_name}")
            except Exception as e:
                logging.error(f"Error processing SMILES '{smiles_str}' for {exp_name}: {e}")
        else:
            logging.warning(f"No valid 'best_smiles' key found in summary for {exp_name}. Summary keys: {list(summary.keys())}")

    if not mols:
        logging.warning("No valid molecules could be generated to draw.")
        return

    logging.info(f"Drawing grid image for {len(mols)} molecules.")
    # draw molecules in a grid
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300), legends=labels)
    img.save(output_dir / "best_molecules.png")
    print(f"Best molecules image saved to {output_dir}/best_molecules.png")


def plot_nfe_comparisons(experiments, results_dict, output_dir):
    """Create plots comparing performance metrics against NFE."""
    logging.info("Plotting performance vs. NFE...")
    
    # prepare data
    plot_data = []
    for exp_name in experiments:
        results = results_dict.get(exp_name)
        if not results or 'nfe' not in results or results['nfe'] <= 0:
            logging.warning(f"Skipping NFE plot for {exp_name}, missing NFE data.")
            continue
        
        improvement = results.get('best_score', 0) - results.get('baseline_score', 0)
        
        # extract algorithm-specific parameters for annotations
        config = results.get('config', {})
        params = ""
        algorithm = results.get('algorithm', 'unknown')
        if algorithm == 'random':
            params = f"trials={config.get('num_trials', '?')}"
        elif algorithm == 'zero_order':
            params = f"steps={config.get('num_steps', '?')}, n={config.get('num_neighbors', '?')}"
        elif algorithm == 'guided':
            params = f"pop={config.get('pop_size', '?')}, gen={config.get('num_generations', '?')}"
        
        plot_data.append({
            'experiment': exp_name,
            'algorithm': algorithm,
            'params': params,
            'nfe': results['nfe'],
            'best_combined': results.get('best_score'),
            'best_sa': results.get('best_sa_score'),
            'best_clogp': results.get('best_clogp_score'),
            'best_qed': results.get('best_qed_score'),  # Include QED if available
            'improvement': improvement
        })
        
    if not plot_data:
        logging.warning("No valid data found for NFE comparison plots.")
        return
        
    df_plot = pd.DataFrame(plot_data)
    
    # define metrics to plot against NFE
    metrics = {
        'best_combined': 'Best Combined Score',
        'improvement': 'Score Improvement over Baseline'
    }
    
    # add individual metrics if they exist in the data
    if 'best_sa' in df_plot.columns and not df_plot['best_sa'].isnull().all():
        metrics['best_sa'] = 'Best SA Score'
    if 'best_clogp' in df_plot.columns and not df_plot['best_clogp'].isnull().all():
        metrics['best_clogp'] = 'Best cLogP Score'
    if 'best_qed' in df_plot.columns and not df_plot['best_qed'].isnull().all():
        metrics['best_qed'] = 'Best QED Score'
    
    # create plots
    num_metrics = len(metrics)
    rows = (num_metrics + 1) // 2 
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6*rows))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    algo_styles = {
        'random': {'marker': 'o', 'color': 'blue'},
        'zero_order': {'marker': 's', 'color': 'green'},
        'guided': {'marker': '^', 'color': 'red'},
        'unknown': {'marker': 'x', 'color': 'gray'}
    }

    for i, (metric_key, metric_label) in enumerate(metrics.items()):
        if i >= len(axes):
            logging.warning(f"Not enough axes for plotting {metric_label}. Skipping.")
            continue
            
        ax = axes[i]
        if metric_key not in df_plot.columns or df_plot[metric_key].isnull().all():
            logging.warning(f"Skipping plot for {metric_label} due to missing data.")
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric_label} vs. NFE')
            continue

        # plot each algorithm separately to assign legend labels correctly
        for algo in df_plot['algorithm'].unique():
            df_algo = df_plot[df_plot['algorithm'] == algo]
            style = algo_styles.get(algo, algo_styles['unknown'])
            scatter = ax.scatter(df_algo['nfe'], df_algo[metric_key],
                       label=algo, marker=style['marker'], color=style['color'], 
                       alpha=0.8, s=80)
            
            # Add annotations for each point
            for _, row in df_algo.iterrows():
                ax.annotate(
                    row['params'],
                    xy=(row['nfe'], row[metric_key]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
        
        ax.set_title(f'{metric_label} vs. NFE')
        ax.set_xlabel('Number of Function Evaluations (NFE)')
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.4)
        ax.legend(title="Algorithm")
        
        # add best fit line if there are enough points per algorithm
        # for algo in df_plot['algorithm'].unique():
        #     df_algo = df_plot[df_plot['algorithm'] == algo]
        #     if len(df_algo) >= 3:  # Only add trend line if we have at least 3 points
        #         try:
        #             x = df_algo['nfe'].values
        #             y = df_algo[metric_key].values
        #             z = np.polyfit(x, y, 1)
        #             p = np.poly1d(z)
        #             x_sorted = np.sort(x)
        #             style = algo_styles.get(algo, algo_styles['unknown'])
        #             ax.plot(x_sorted, p(x_sorted), '--', color=style['color'], alpha=0.5)
        #         except Exception as e:
        #             logging.warning(f"Could not add trend line for {algo}: {e}")

    # Remove unused subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = output_dir / "performance_vs_nfe.png"
    plt.savefig(plot_path)
    plt.close(fig)
    
    # Create a separate plot showing NFE efficiency
    plt.figure(figsize=(10, 6))
    
    # Calculate efficiency (improvement per NFE)
    df_plot['efficiency'] = df_plot['improvement'] / df_plot['nfe']
    
    # Sort by efficiency
    df_plot = df_plot.sort_values('efficiency', ascending=False)
    
    # Create bar chart of efficiency
    bars = plt.bar(range(len(df_plot)), df_plot['efficiency'], color=[algo_styles.get(algo, algo_styles['unknown'])['color'] for algo in df_plot['algorithm']])
    
    # Add labels
    plt.xlabel('Experiment')
    plt.ylabel('Improvement per NFE')
    plt.title('Efficiency of Different Search Strategies')
    
    # Add x-tick labels with algorithm and params
    plt.xticks(range(len(df_plot)), [f"{row['algorithm']}\n{row['params']}" for _, row in df_plot.iterrows()], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.0001,
            f"{height:.5f}",
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    plt.tight_layout()
    efficiency_path = output_dir / "nfe_efficiency.png"
    plt.savefig(efficiency_path)
    plt.close()
    
    logging.info(f"Saved Performance vs. NFE comparison plots to {output_dir}")


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
    
    # plot NFE comparisons
    plot_nfe_comparisons(list(results_dict.keys()), results_dict, output_dir)
    
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