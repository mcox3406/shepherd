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
        
        # Filter out any zero scores (should be rare in the progression data)
        if len(scores) > 0 and min(scores) == 0:
            scores = [s for s in scores if s > 0]
            logging.info(f"Filtered out zero scores from progression data for {exp_name}")
        
        if not scores:
            logging.warning(f"No valid scores left after filtering zeros for {exp_name}")
            continue
        
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
        if results and 'baseline_score' in results and results['baseline_score'] > 0:
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
        
        # Skip if either score is zero
        if results['best_score'] <= 0 or results['baseline_score'] <= 0:
            logging.warning(f"Skipping improvement calculation for {exp_name} due to zero scores")
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
    bar_width = 0.6  # Wider bars
    plt.figure(figsize=(max(12, len(labels)*1.5), 6))  # Adjust width based on number of bars
    
    bars = plt.bar(x_pos, improvements, width=bar_width)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Score Improvement over Baseline')
    plt.title('Comparison of Search Algorithm Improvements')
    plt.grid(axis='y', alpha=0.3)
    
    # Set x-axis ticks and labels with proper rotation if needed
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
        
        # Extract algorithm-specific parameters for annotations
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
    
    # Add individual metrics if they exist in the data
    if 'best_sa' in df_plot.columns and not df_plot['best_sa'].isnull().all():
        metrics['best_sa'] = 'Best SA Score'
    if 'best_clogp' in df_plot.columns and not df_plot['best_clogp'].isnull().all():
        metrics['best_clogp'] = 'Best cLogP Score'
    if 'best_qed' in df_plot.columns and not df_plot['best_qed'].isnull().all():
        metrics['best_qed'] = 'Best QED Score'
    
    # create plots
    num_metrics = len(metrics)
    rows = (num_metrics + 1) // 2  # Calculate rows needed
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6*rows))
    if rows == 1:
        axes = np.array([axes])  # Ensure axes is always 2D
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


def plot_property_distributions(experiments, base_dir, output_dir):
    """
    Plot the distributions of molecular properties across all generated molecules for each experiment.
    
    This function reads the all_molecules_log.csv files for each experiment and creates
    histogram/KDE plots for SA score, cLogP, QED, and combined score distributions.
    """
    logging.info("Plotting property distributions across experiments...")
    
    # Properties to analyze
    properties = ['sa_score', 'clogp_score', 'qed_score', 'combined_score']
    property_labels = {
        'sa_score': 'Synthetic Accessibility Score',
        'clogp_score': 'cLogP Score',
        'qed_score': 'QED (Drug-likeness) Score',
        'combined_score': 'Combined Score'
    }
    
    # rescaling functions to convert normalized scores back to original scales for visualization
    def rescale_sa_score(normalized_score):
        """Convert normalized SA score [0,1] back to original [1,10] scale (inverted for consistency)"""
        # s_norm = 1 - (s_raw - 1)/9, so s_raw = 1 + 9 * (1 - s_norm)
        return 1 + 9 * (1 - normalized_score)
    
    def rescale_clogp_score(normalized_score):
        """Convert normalized cLogP score [0,1] back to original [-1,15] scale"""
        # default range is [-1, 15]
        return normalized_score * (15 - (-1)) + (-1)
    
    # no rescaling needed for QED as it's already in [0,1]
    
    rescaling_functions = {
        'sa_score': rescale_sa_score,
        'clogp_score': rescale_clogp_score,
        'qed_score': lambda x: x,  # identity function, no rescaling
        'combined_score': lambda x: x  # identity function, no rescaling
    }
    
    # Axis labels for rescaled properties
    rescaled_labels = {
        'sa_score': 'SA Score (1=easy to 10=difficult)',
        'clogp_score': 'cLogP',
        'qed_score': 'QED',
        'combined_score': 'Combined Score'
    }
    
    # Load data for each experiment
    experiment_data = {}
    for exp_name in experiments:
        exp_dir = Path(base_dir) / exp_name
        log_file = exp_dir / "all_molecules_log.csv"
        
        if not log_file.exists():
            logging.warning(f"No molecule log found for experiment {exp_name}, skipping.")
            continue
            
        try:
            df = pd.read_csv(log_file)
            
            # Filter out molecules with score of 0 (failed to parse)
            initial_count = len(df)
            df = df[(df['sa_score'] > 0) & (df['clogp_score'] > 0) & (df['combined_score'] > 0)]
            if 'qed_score' in df.columns:
                df = df[df['qed_score'] > 0]
            filtered_count = len(df)
            
            if filtered_count < initial_count:
                logging.info(f"Filtered out {initial_count - filtered_count} molecules with zero scores from {exp_name}")
            
            if filtered_count == 0:
                logging.warning(f"No valid molecules left after filtering for {exp_name}, skipping.")
                continue
            
            # Add algorithm information to the dataframe
            # Extract algorithm and key parameters for legend
            parts = exp_name.split('_')
            if len(parts) > 0:
                algo = parts[0]
                df['algorithm'] = algo
                
                # Try to extract key parameters based on algorithm type
                if algo == 'random' and 'trials' in exp_name:
                    for part in parts:
                        if part.startswith('nfe'):
                            df['nfe_target'] = part[3:]  # Extract the NFE target
                
                elif algo == 'zero' or algo == 'zero_order':
                    params = []
                    for part in parts:
                        if part.startswith('s'):  # steps
                            params.append(f"steps={part[1:]}")
                        elif part.startswith('n') and not part.startswith('nfe'):  # neighbors
                            params.append(f"nbrs={part[1:]}")
                    if params:
                        df['params'] = ', '.join(params)
                
                elif algo == 'guided':
                    params = []
                    for part in parts:
                        if part.startswith('p'):  # population
                            params.append(f"pop={part[1:]}")
                        elif part.startswith('g'):  # generations
                            params.append(f"gen={part[1:]}")
                    if params:
                        df['params'] = ', '.join(params)
            
            # Create a display name for the experiment
            if 'params' in df.columns:
                df['experiment_display'] = df['algorithm'] + ' (' + df['params'] + ')'
            elif 'nfe_target' in df.columns:
                df['experiment_display'] = df['algorithm'] + ' (NFE=' + df['nfe_target'] + ')'
            else:
                df['experiment_display'] = exp_name
                
            # Add rescaled properties for visualization
            for prop in properties:
                if prop in df.columns:
                    rescale_func = rescaling_functions[prop]
                    df[f'{prop}_rescaled'] = df[prop].apply(rescale_func)
                
            experiment_data[exp_name] = df
            logging.info(f"Loaded {len(df)} valid molecules from {exp_name}")
        except Exception as e:
            logging.error(f"Error loading data for {exp_name}: {e}")
    
    if not experiment_data:
        logging.warning("No valid experiment data found for property distribution analysis.")
        return
    
    # Create a single combined dataframe with experiment information
    all_data = pd.concat(experiment_data.values(), ignore_index=True)
    
    # Create distribution plots for each property (using rescaled values)
    for prop in properties:
        if prop not in all_data.columns:
            logging.warning(f"Property {prop} not found in experiment data.")
            continue
            
        # Get the rescaled property name
        rescaled_prop = f'{prop}_rescaled'
        
        # Create a figure large enough to show all distributions clearly
        plt.figure(figsize=(12, 8))
        
        # Use seaborn to create the KDE plot with histograms
        ax = sns.histplot(
            data=all_data, 
            x=rescaled_prop, 
            hue='experiment_display',
            kde=True,
            element="step",
            common_norm=False,  # Each distribution has its own normalization
            stat="density",     # Show density rather than count
            alpha=0.4,          # Make histograms semi-transparent
            linewidth=2,        # Line width for KDE
            palette="viridis"   # Color palette
        )
        
        # Improve the legend
        handles, labels = ax.get_legend_handles_labels()
        
        # Only create legend if we have labels
        if handles and labels:
            # Shorten legend labels to prevent overcrowding
            short_legend_labels = []
            for label in labels:
                if '(' in label:
                    algo, params = label.split('(', 1)
                    # Take just the first parameter 
                    if ',' in params:
                        first_param = params.split(',')[0]
                        short_label = f"{algo}({first_param})"
                    else:
                        short_label = f"{algo}({params}"
                else:
                    short_label = label
                short_legend_labels.append(short_label)
                
            ax.legend(
                handles=handles, 
                labels=short_legend_labels, 
                title="Experiment",
                loc="upper right", 
                frameon=True, 
                framealpha=0.9,
                fontsize=9
            )
        
        # add labels and title
        plt.xlabel(rescaled_labels.get(prop, prop))
        plt.ylabel("Density")
        plt.title(f"Distribution of {property_labels.get(prop, prop)} Across Experiments")
        plt.grid(alpha=0.3)
        
        # for SA Score, invert the x-axis to show 1 (easy) on the left and 10 (difficult) on the right
        # if prop == 'sa_score':
        #     xlim = plt.xlim()
        #     plt.xlim(min(1, xlim[0]), max(10, xlim[1]))
        
        # for cLogP, set reasonable limits
        # if prop == 'clogp_score':
        #     plt.xlim(-2, 16)
            
        # Save the figure
        plt.tight_layout()
        output_path = output_dir / f"{prop}_distribution_comparison.png"
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved {prop} distribution plot to {output_path}")
    
    # Create a pairplot for the main properties if there are multiple experiments
    if len(experiment_data) > 1:
        try:
            # Limit to a subset of rows if there are too many molecules
            max_rows_per_exp = 1000
            sample_data = []
            
            for exp_name, df in experiment_data.items():
                if len(df) > max_rows_per_exp:
                    sample_df = df.sample(max_rows_per_exp, random_state=42)
                else:
                    sample_df = df
                sample_data.append(sample_df)
            
            sample_all_data = pd.concat(sample_data, ignore_index=True)
            
            # List of rescaled properties for the pairplot (exclude combined score)
            pairplot_props = [f'{p}_rescaled' for p in properties if p != 'combined_score' and f'{p}_rescaled' in sample_all_data.columns]
            
            if len(pairplot_props) >= 2:  # Need at least 2 properties for a pairplot
                # Create the pairplot
                g = sns.pairplot(
                    data=sample_all_data,
                    vars=pairplot_props,
                    hue='experiment_display',
                    diag_kind='kde',
                    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'},
                    diag_kws={'alpha': 0.8},
                    palette="viridis"
                )
                
                # Update axis labels to show original scale
                for i, var in enumerate(pairplot_props):
                    original_prop = var.replace('_rescaled', '')
                    for j in range(len(pairplot_props)):
                        if i == j:  # Diagonal
                            g.axes[i, i].set_xlabel(rescaled_labels.get(original_prop, original_prop))
                        else:  # Off-diagonal
                            if i < len(g.axes) and j < len(g.axes[i]):
                                g.axes[i, j].set_xlabel(rescaled_labels.get(pairplot_props[j].replace('_rescaled', ''), pairplot_props[j]))
                            if j < len(g.axes) and i < len(g.axes[j]):
                                g.axes[j, i].set_ylabel(rescaled_labels.get(pairplot_props[i].replace('_rescaled', ''), pairplot_props[i]))
                
                # Create shorter labels for the legend
                try:
                    if hasattr(g, '_legend_data') and g._legend_data:
                        # In newer versions of seaborn, handles are keys and labels are values
                        if isinstance(list(g._legend_data.keys())[0], plt.matplotlib.artist.Artist):
                            handles = list(g._legend_data.keys())
                            labels = list(g._legend_data.values())
                        else:
                            # In some versions, it might be reversed
                            labels = list(g._legend_data.keys())
                            handles = list(g._legend_data.values())
                        
                        # Shorten the labels
                        short_labels = []
                        for label in labels:
                            if isinstance(label, str) and '(' in label:
                                algo, params = label.split('(', 1)
                                if ',' in params:
                                    first_param = params.split(',')[0]
                                    short_label = f"{algo}({first_param})"
                                else:
                                    short_label = f"{algo}({params}"
                            else:
                                short_label = str(label)
                            short_labels.append(short_label)
                        
                        # Create a new legend with shortened labels
                        if hasattr(g, '_legend') and g._legend:
                            g._legend.remove()
                        g.fig.legend(handles, short_labels, title="Experiment", 
                                    loc='upper right', bbox_to_anchor=(0.99, 0.99),
                                    frameon=True, framealpha=0.9, fontsize=9)
                    elif hasattr(g, '_legend'):
                        # Improve the default legend if we can't create a custom one
                        g._legend.set_title("Experiment")
                except Exception as e:
                    logging.warning(f"Could not customize pairplot legend: {e}")
                    # Default legend behavior
                    if hasattr(g, '_legend'):
                        g._legend.set_title("Experiment")
                
                # Save the figure
                pairplot_path = output_dir / "property_pairplot.png"
                plt.savefig(pairplot_path, dpi=150)
                plt.close()
                logging.info(f"Saved property pairplot to {pairplot_path}")
        except Exception as e:
            logging.error(f"Error creating pairplot: {e}")

    # create a box plot to compare distributions across experiments
    plt.figure(figsize=(14, 10))
    
    # set up subplots for each property
    num_props = len([p for p in properties if f'{p}_rescaled' in all_data.columns])
    fig, axes = plt.subplots(num_props, 1, figsize=(14, 4*num_props))
    
    # handle case of single property
    if num_props == 1:
        axes = [axes]
    
    # create box plots for each property
    for i, prop in enumerate([p for p in properties if f'{p}_rescaled' in all_data.columns]):
        rescaled_prop = f'{prop}_rescaled'
        
        sns.boxplot(
            data=all_data,
            x='experiment_display',
            y=rescaled_prop,
            ax=axes[i],
            hue='experiment_display',
            legend=False,
            palette="viridis"
        )
        
        # add individual points
        sns.stripplot(
            data=all_data,
            x='experiment_display',
            y=rescaled_prop,
            ax=axes[i],
            size=3,
            color='black',
            alpha=0.3,
            jitter=True
        )
        
        axes[i].set_title(f"Distribution of {property_labels.get(prop, prop)} by Experiment")
        axes[i].set_ylabel(rescaled_labels.get(prop, prop))
        axes[i].set_xlabel("")
        
        # create shorter experiment labels for the x-axis
        if i == num_props - 1:  # only relabel the last subplot
            xlabels = axes[i].get_xticklabels()
            short_labels = []
            for label in xlabels:
                text = label.get_text()
                # shorten the label: keep algorithm name and first parameter only
                if '(' in text:
                    algo, params = text.split('(', 1)
                    params = params.split(',')[0] + ')'  # take only the first parameter
                    short_label = f"{algo}({params}"
                else:
                    short_label = text
                short_labels.append(short_label)
            
            # get the current tick positions and set them explicitly before setting labels
            tick_locs = axes[i].get_xticks()
            if len(short_labels) == len(tick_locs):
                axes[i].set_xticks(tick_locs)
                axes[i].set_xticklabels(short_labels, rotation=45, ha='right')
            else:
                # if counts don't match, let matplotlib handle it automatically
                logging.warning(f"Number of labels ({len(short_labels)}) doesn't match tick positions ({len(tick_locs)})")
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        else:
            # hide x-tick labels for all but the last subplot
            axes[i].set_xticklabels([])
            axes[i].set_xlabel("")
        
        # for certain properties, set specific y-axis limits
        # if prop == 'sa_score':
        #     axes[i].set_ylim(1, 10)
        # elif prop == 'clogp_score':
        #     axes[i].set_ylim(-2, 16)
        # elif prop == 'qed_score':
        #     axes[i].set_ylim(0, 1)
    
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0, rect=[0, 0.05, 1, 0.95])
    boxplot_path = output_dir / "property_boxplots.png"
    plt.savefig(boxplot_path)
    plt.close()
    logging.info(f"Saved property boxplots to {boxplot_path}")


def plot_failure_rates_by_algorithm(experiments, base_dir, output_dir):
    """
    Plot the fraction of molecules with 0 scores (failed parsing) by search algorithm type.
    
    Args:
        experiments: List of experiment names
        base_dir: Base directory containing the experiment results
        output_dir: Directory to save the analysis results
    """
    logging.info("Analyzing molecule parsing failure rates by algorithm type...")
    
    # data structure to store failure rates
    algorithm_failure_data = {
        'random': {'total': 0, 'failed': 0, 'experiments': 0},
        'zero_order': {'total': 0, 'failed': 0, 'experiments': 0},
        'guided': {'total': 0, 'failed': 0, 'experiments': 0},
        'other': {'total': 0, 'failed': 0, 'experiments': 0}
    }
    
    # process each experiment
    for exp_name in experiments:
        exp_dir = Path(base_dir) / exp_name
        log_file = exp_dir / "all_molecules_log.csv"
        
        if not log_file.exists():
            logging.warning(f"No molecule log found for experiment {exp_name}, skipping.")
            continue
        
        # determine algorithm type from experiment name
        algorithm_type = 'other'
        if 'random' in exp_name.lower():
            algorithm_type = 'random'
        elif 'zero' in exp_name.lower() or 'zo' in exp_name.lower():
            algorithm_type = 'zero_order'
        elif 'guided' in exp_name.lower():
            algorithm_type = 'guided'
        
        try:
            df = pd.read_csv(log_file)
            
            total_molecules = len(df)
            
            # count molecules with 0 scores (failed parsing)
            failed_molecules = 0
            if 'sa_score' in df.columns:
                failed_molecules += sum(df['sa_score'] == 0)
            # if 'clogp_score' in df.columns:
            #     failed_molecules += sum(df['clogp_score'] == 0)
            # if 'qed_score' in df.columns:
            #     failed_molecules += sum(df['qed_score'] == 0)
            
            # take the min count as molecules might fail multiple verifiers
            # failed_molecules = min(failed_molecules, total_molecules)
            
            # add to accumulated data
            algorithm_failure_data[algorithm_type]['total'] += total_molecules
            algorithm_failure_data[algorithm_type]['failed'] += failed_molecules
            algorithm_failure_data[algorithm_type]['experiments'] += 1
            
            logging.info(f"Experiment {exp_name} ({algorithm_type}): {failed_molecules}/{total_molecules} failed molecules ({failed_molecules/total_molecules*100:.2f}%)")
            
        except Exception as e:
            logging.error(f"Error processing failure rates for {exp_name}: {e}")
    
    # Calculate failure rates
    failure_rates = []
    algorithm_labels = []
    success_rates = []
    exp_counts = []
    
    for algo, data in algorithm_failure_data.items():
        if data['total'] > 0:
            # Calculate failure rate
            failure_rate = data['failed'] / data['total']
            success_rate = 1 - failure_rate
            
            failure_rates.append(failure_rate)
            success_rates.append(success_rate)
            algorithm_labels.append(algo.replace('_', ' ').title())
            exp_counts.append(data['experiments'])
            
            logging.info(f"{algo}: {failure_rate*100:.2f}% failure rate across {data['experiments']} experiments ({data['failed']}/{data['total']} molecules)")
    
    if not algorithm_labels:
        logging.warning("No valid algorithm data found for failure rate analysis.")
        return
    
    # stacked bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.6
    
    x_pos = np.arange(len(algorithm_labels))
    
    # plot the successful molecules (bottom part of the stack)
    success_bars = plt.bar(x_pos, success_rates, bar_width, 
                          label='Successfully Parsed', 
                          color='mediumseagreen', edgecolor='darkgreen')
    
    # plot the failed molecules (top part of the stack)
    failure_bars = plt.bar(x_pos, failure_rates, bar_width,
                          bottom=success_rates,
                          label='Failed to Parse', 
                          color='lightcoral', edgecolor='darkred')
    
    # add experiment count as text on bars
    for i, (failure, success, count) in enumerate(zip(failure_rates, success_rates, exp_counts)):
        # add failure rate percentage text
        plt.text(i, success + failure/2, f"{failure*100:.1f}%", 
                ha='center', va='center', color='black', fontweight='bold')
        
        # add experiment count at the bottom
        plt.text(i, -0.05, f"n={count} exps", ha='center', va='top')
    
    # plt.xlabel('Search Algorithm')
    plt.ylabel('Fraction of Molecules')
    plt.title('Molecule Parsing Success/Failure Rate by Search Algorithm')
    plt.xticks(x_pos, algorithm_labels)
    plt.ylim(0, 1.1)  # Make room for labels
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "failure_rate_by_algorithm.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved failure rate by algorithm plot to {plot_path}")
    
    # second plot showing exact molecule counts
    plt.figure(figsize=(10, 6))
    
    algorithms = []
    successful_counts = []
    failed_counts = []
    
    for algo, data in algorithm_failure_data.items():
        if data['total'] > 0:
            algorithms.append(algo.replace('_', ' ').title())
            successful_counts.append(data['total'] - data['failed'])
            failed_counts.append(data['failed'])
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    success_bars = ax.bar(x - width/2, successful_counts, width, label='Successfully Parsed', color='mediumseagreen', edgecolor='darkgreen')    
    failure_bars = ax.bar(x + width/2, failed_counts, width, label='Failed to Parse', color='lightcoral', edgecolor='darkred')
    
    for i, (success_count, fail_count) in enumerate(zip(successful_counts, failed_counts)):
        ax.text(i - width/2, success_count, str(success_count), ha='center', va='bottom')
        ax.text(i + width/2, fail_count, str(fail_count), ha='center', va='bottom')
    
    # ax.set_xlabel('Search Algorithm')
    ax.set_ylabel('Number of Molecules')
    ax.set_title('Number of Successfully Parsed vs. Failed Molecules by Search Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    count_plot_path = output_dir / "molecule_counts_by_algorithm.png"
    plt.savefig(count_plot_path)
    plt.close()
    logging.info(f"Saved molecule counts by algorithm plot to {count_plot_path}")


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
    
    # plot property distributions (new)
    plot_property_distributions(list(results_dict.keys()), base_dir, output_dir)
    
    # plot failure rates by algorithm (new)
    plot_failure_rates_by_algorithm(list(results_dict.keys()), base_dir, output_dir)
    
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