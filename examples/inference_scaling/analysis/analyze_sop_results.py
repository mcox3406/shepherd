"""
Analyze and visualize SearchOverPaths experiment results.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any
import re
import argparse
import glob # Added for glob functionality

def load_pickle_results(file_path: Path) -> Dict[str, Any]:
    """
    Load a pickle file containing SearchOverPaths experiment results.
    
    Args:
        file_path: Path to the results.pkl file
        
    Returns:
        Dictionary containing the experiment results
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def extract_NM_from_path(exp_path: Path) -> Tuple[int, int]:
    """
    Extract N and M values from the experiment path.
    
    Args:
        exp_path: Path to the experiment directory
        
    Returns:
        Tuple of (N, M) values
    """
    # Pattern to match N and M values in path name
    pattern = r"search_over_paths_N(\d+)_M(\d+)"
    match = re.search(pattern, str(exp_path))
    
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_experiment_results(base_dirs: List[str]) -> pd.DataFrame:
    """
    Load all experiment results from the specified directories.
    
    Args:
        base_dirs: List of base directories containing experiment results
        
    Returns:
        DataFrame containing the experiment results
    """
    results = []
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        
        # Check if base_path exists
        if not base_path.exists():
            print(f"Directory not found: {base_path}")
            continue
        
        # Find all experiment directories matching the pattern
        pattern = r"search_over_paths_N\d+_M\d+"
        exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and re.match(pattern, d.name)]
        
        for exp_dir in exp_dirs:
            # Find results.pkl in the experiment directory
            results_path = exp_dir / "results.pkl"
            if not results_path.exists():
                print(f"No results.pkl found in {exp_dir}")
                continue
                
            # Load the results
            exp_data = load_pickle_results(results_path)
            if not exp_data:
                continue
                
            # Extract N and M from directory name
            N, M = extract_NM_from_path(exp_dir)
            if N is None or M is None:
                print(f"Could not extract N and M from {exp_dir}")
                continue
                
            # Extract key information from the results based on SearchOverPaths structure
            metadata = exp_data.get('metadata', {})
            
            # Determine objective type from the directory path
            objective_type = 'multi' if 'multi_objective' in str(exp_dir) else 'single'
            
            # For SearchOverPaths, we have specific fields in metadata
            result = {
                'experiment_name': exp_dir.name,
                'objective_type': objective_type,
                'N': N,
                'M': M,
                'nfe': metadata.get('nfe', 0),
                'total_time': metadata.get('total_time', 0),
                'num_paths': metadata.get('num_paths', N),
                'paths_width': metadata.get('paths_width', M)
            }
            
            # Extract scores - best_final_score is in metadata, but best_overall_score is in results['best_score']
            result['best_final_score'] = metadata.get('best_final_score', 0)
            result['best_overall_score'] = exp_data.get('best_score', 0)  # Corrected location
            
            # Add other useful metadata
            result['num_iterations'] = metadata.get('num_iterations', 0)
            result['nfe_per_path'] = metadata.get('nfe_per_path', 0)
            
            # Add efficiency metrics
            if result['nfe'] > 0:
                result['score_per_nfe_final'] = result['best_final_score'] / result['nfe']
                result['score_per_nfe_overall'] = result['best_overall_score'] / result['nfe']
            else:
                result['score_per_nfe_final'] = 0
                result['score_per_nfe_overall'] = 0
                
            # Extract SMILES if available
            result['best_smiles'] = exp_data.get('best_smiles', '')
            
            results.append(result)
            
    return pd.DataFrame(results)

def process_random_search_data(base_experiment_dir: str) -> Dict[str, Dict[str, pd.Series]]:
    """
    Parses random search CSV files, calculates cumulative maximum of 'combined_score'
    and efficiency (cummax_score / NFE). Returns mean and std dev for these metrics.

    Args:
        base_experiment_dir: The base directory for random experiments 
                             (e.g., "examples/inference_scaling/inference_scaling_experiments/random").
        
    Returns:
        A dictionary where keys are objective types ("single_objective", "multi_objective")
        and values are dictionaries containing pandas Series for:
        "mean_cummax_score", "std_cummax_score", 
        "mean_efficiency", "std_efficiency".
    """
    objective_types = ["single_objective", "multi_objective"]
    all_processed_data = {}

    for objective_type in objective_types:
        all_runs_cummax_scores_list = []
        all_runs_efficiency_list = []
        max_nfe_overall = 0 # To align indices later if necessary

        for i in range(4):  # job0 to job3
            for j in range(4):  # repeat_0 to repeat_3
                search_pattern = os.path.join(
                    base_experiment_dir, 
                    objective_type, 
                    f"job{i}", 
                    f"repeat_{j}", 
                    "*", # For the timestamped directory
                    "all_molecules_log.csv"
                )
                file_paths = glob.glob(search_pattern)
                
                if not file_paths:
                    print(f"Warning: No CSV file found for {objective_type}, job{i}, repeat_{j} with pattern {search_pattern}")
                    continue
                
                file_path = file_paths[0]
                try:
                    df = pd.read_csv(file_path)
                    if 'iteration' not in df.columns or 'combined_score' not in df.columns:
                        print(f"Warning: CSV {file_path} is missing 'iteration' or 'combined_score' column.")
                        continue
                    
                    df = df.sort_values(by='iteration').reset_index(drop=True)
                    df['cummax_score'] = df['combined_score'].cummax()
                    
                    # Use 'iteration' as NFE directly. For efficiency, handle iteration 0.
                    df['nfe'] = df['iteration'] # NFE is the iteration number
                    df['efficiency'] = df['cummax_score'] / (df['nfe'] + 1e-9) # Add small epsilon to avoid division by zero warning, will result in inf or large number
                    df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], np.nan) # Replace inf with NaN
                    
                    # Store series indexed by 'nfe' (iteration)
                    run_cummax_scores = df.set_index('nfe')['cummax_score']
                    run_efficiency = df.set_index('nfe')['efficiency']
                    
                    all_runs_cummax_scores_list.append(run_cummax_scores)
                    all_runs_efficiency_list.append(run_efficiency)

                    if df['nfe'].max() > max_nfe_overall:
                        max_nfe_overall = df['nfe'].max()
                        
                except pd.errors.EmptyDataError:
                    print(f"Warning: CSV file {file_path} is empty.")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        if not all_runs_cummax_scores_list:
            print(f"No data processed for random search {objective_type}.")
            all_processed_data[objective_type] = {
                "mean_cummax_score": pd.Series(dtype=float),
                "std_cummax_score": pd.Series(dtype=float),
                "mean_efficiency": pd.Series(dtype=float),
                "std_efficiency": pd.Series(dtype=float)
            }
            continue

        # Create a common index from 0 to max_nfe_overall
        common_index = pd.Index(range(max_nfe_overall + 1), name="nfe")

        # Align and concatenate cummax scores
        aligned_cummax_scores = [
            s.reindex(common_index).ffill().bfill() for s in all_runs_cummax_scores_list
        ]
        combined_cummax_df = pd.concat(aligned_cummax_scores, axis=1)
        
        # Align and concatenate efficiency scores
        aligned_efficiency = [
            s.reindex(common_index).ffill().bfill() for s in all_runs_efficiency_list
        ]
        combined_efficiency_df = pd.concat(aligned_efficiency, axis=1)

        # Calculate mean and std
        mean_cummax_score = combined_cummax_df.mean(axis=1)
        std_cummax_score = combined_cummax_df.std(axis=1)
        mean_efficiency = combined_efficiency_df.mean(axis=1)
        std_efficiency = combined_efficiency_df.std(axis=1)
        
        all_processed_data[objective_type] = {
            "mean_cummax_score": mean_cummax_score,
            "std_cummax_score": std_cummax_score,
            "mean_efficiency": mean_efficiency,
            "std_efficiency": std_efficiency
        }
        print(f"Processed random search data for {objective_type}")

    return all_processed_data

def create_score_vs_nfe_by_N_plot(df: pd.DataFrame, score_type: str = 'final', 
                            objective_type: str = None, random_mean_scores=None, random_std_scores=None):
    """
    Create a line plot of score vs NFE with separate lines for different N values.
    
    Args:
        df: DataFrame containing the experiment results
        score_type: Either 'final' or 'overall' to specify which score to plot
        objective_type: Filter by objective type ('single', 'multi', or None for all)
        random_mean_scores: Mean scores from random search for comparison
        random_std_scores: Standard deviation scores from random search for comparison
    """
    if objective_type:
        filtered_df = df[df['objective_type'] == objective_type]
        if filtered_df.empty:
            print(f"No data for objective_type={objective_type}")
            return None
    else:
        filtered_df = df

    plt.figure(figsize=(10, 6))
    
    score_key = f'best_{score_type}_score'
    
    # Group by N value
    unique_n_values = sorted(filtered_df['N'].unique())
    colors = sns.color_palette('husl', n_colors=len(unique_n_values))
    
    # Plot a line for each N value (SearchOverPaths)
    for i, n_val in enumerate(unique_n_values):
        n_data = filtered_df[filtered_df['N'] == n_val].sort_values('nfe')
        
        if len(n_data) > 0:
            plt.plot(
                n_data['nfe'],
                n_data[score_key],
                marker='o',
                linestyle='-',
                label=f'SOP N={n_val}', # Clarify SOP
                color=colors[i]
            )
            
            # Add M value annotations next to each point
            for _, row in n_data.iterrows():
                plt.annotate(
                    f"M={row['M']}",
                    (row['nfe'], row[score_key]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8
                )
    
    if (random_mean_scores is not None and not random_mean_scores.empty and
        random_std_scores is not None and not random_std_scores.empty):
        plt.plot(random_mean_scores.index, random_mean_scores, linestyle='--', label='Random Search Mean', color='grey')
        plt.fill_between(random_mean_scores.index, random_mean_scores - random_std_scores, random_mean_scores + random_std_scores, alpha=0.2, color='grey', label='Random Search Std Dev')
    
    plt.xlabel('Number of Function Evaluations (NFE)')
    plt.ylabel(f'Best {score_type.capitalize()} Score')
    objective_str = f" ({objective_type} objective)" if objective_type else ""
    plt.title(f'Best {score_type.capitalize()} Score vs NFE by N Value{objective_str}')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Configuration') # Generalize legend title
    plt.tight_layout()
    return plt.gcf()

def create_score_vs_nfe_by_M_plot(df: pd.DataFrame, score_type: str = 'final', 
                            objective_type: str = None, random_mean_scores=None, random_std_scores=None):
    """
    Create a line plot of score vs NFE with separate lines for different M values.
    
    Args:
        df: DataFrame containing the experiment results
        score_type: Either 'final' or 'overall' to specify which score to plot
        objective_type: Filter by objective type ('single', 'multi', or None for all)
        random_mean_scores: Mean scores from random search for comparison
        random_std_scores: Standard deviation scores from random search for comparison
    """
    if objective_type:
        filtered_df = df[df['objective_type'] == objective_type]
        if filtered_df.empty:
            print(f"No data for objective_type={objective_type}")
            return None
    else:
        filtered_df = df

    plt.figure(figsize=(10, 6))
    score_key = f'best_{score_type}_score'
    
    # Group by M value
    unique_m_values = sorted(filtered_df['M'].unique())
    # Use 'husl' color palette, consistent with N-plot
    colors = sns.color_palette('husl', n_colors=len(unique_m_values))
    
    for i, m_val in enumerate(unique_m_values):
        m_data = filtered_df[filtered_df['M'] == m_val].sort_values('nfe')
        if len(m_data) > 0:
            plt.plot(
                m_data['nfe'],
                m_data[score_key],
                marker='o',         # Change marker to 'o'
                linestyle='-',       # Change linestyle to '-'
                label=f'SOP M={m_val}',
                color=colors[i]
            )
            
            # Add N value annotations next to each point
            for _, row in m_data.iterrows():
                plt.annotate(
                    f"N={row['N']}",
                    (row['nfe'], row[score_key]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8
                )
                
    if (random_mean_scores is not None and not random_mean_scores.empty and
        random_std_scores is not None and not random_std_scores.empty):
        plt.plot(random_mean_scores.index, random_mean_scores, linestyle=':', label='Random Search Mean', color='black')
        plt.fill_between(random_mean_scores.index, random_mean_scores - random_std_scores, random_mean_scores + random_std_scores, alpha=0.2, color='black', label='Random Search Std Dev')
        
    plt.xlabel('Number of Function Evaluations (NFE)')
    plt.ylabel(f'Best {score_type.capitalize()} Score')
    objective_str = f" ({objective_type} objective)" if objective_type else ""
    plt.title(f'Best {score_type.capitalize()} Score vs NFE by Path Width (M){objective_str}')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Configuration') # Generalize legend title
    plt.tight_layout()
    return plt.gcf()

def create_efficiency_line_plot(df: pd.DataFrame, score_type: str = 'final',
                          objective_type: str = None, random_mean_efficiency=None, random_std_efficiency=None):
    """
    Create a line plot showing score per NFE vs NFE with lines for different N values.
    Args:
        df: DataFrame containing the experiment results
        score_type: Either 'final' or 'overall' to specify which score to plot
        objective_type: Filter by objective type ('single', 'multi', or None for all)
        random_mean_efficiency: Mean efficiency from random search for comparison
        random_std_efficiency: Standard deviation efficiency from random search for comparison
    """
    if objective_type:
        filtered_df = df[df['objective_type'] == objective_type]
        if filtered_df.empty:
            print(f"No data for SOP efficiency plot, objective_type={objective_type}")
            # Still proceed if random data might be plotted, or return None if SOP must exist
            # For now, let SOP data be optional if random data is to be plotted.
    else:
        filtered_df = df # May be empty if df was empty initially
    
    plt.figure(figsize=(10, 6))
    
    # Plot SOP data if available
    if not filtered_df.empty:
        score_key = f'score_per_nfe_{score_type}'
        unique_n_values = sorted(filtered_df['N'].unique())
        colors = sns.color_palette('husl', n_colors=len(unique_n_values))
        
        for i, n_val in enumerate(unique_n_values):
            n_data = filtered_df[filtered_df['N'] == n_val].sort_values('nfe')
            if len(n_data) > 0:
                plt.plot(n_data['nfe'], n_data[score_key], marker='o', linestyle='-', label=f'SOP N={n_val}', color=colors[i])
                for _, row in n_data.iterrows():
                    plt.annotate(f"M={row['M']}", (row['nfe'], row[score_key]), xytext=(5, 0), textcoords='offset points', fontsize=8)
    else:
        print(f"No SOP data to plot for efficiency, objective_type={objective_type}")

    # Plot Random Search Efficiency if available, filtering initial high values
    if (random_mean_efficiency is not None and not random_mean_efficiency.empty and
        random_std_efficiency is not None and not random_std_efficiency.empty):
        
        # Define a threshold for very high efficiency values often seen at NFE=0
        efficiency_threshold = 1.5 # Adjusted as per typical plot scales; user mentioned "~1.0"
        
        # Find the first index (NFE) where mean efficiency is at or below the threshold
        sensible_start_index = random_mean_efficiency[random_mean_efficiency <= efficiency_threshold].first_valid_index()
        
        plot_random_mean_eff = random_mean_efficiency
        plot_random_std_eff = random_std_efficiency
        
        if sensible_start_index is not None:
            if sensible_start_index > plot_random_mean_eff.index.min(): # Check if any data is actually skipped
                print(f"[{objective_type}] Skipping initial random search efficiency data up to NFE {sensible_start_index-1} as it's above threshold {efficiency_threshold}")
            plot_random_mean_eff = random_mean_efficiency.loc[sensible_start_index:]
            plot_random_std_eff = random_std_efficiency.loc[sensible_start_index:]
        else:
            # All values are above threshold, so random efficiency plot will be empty or not shown
            print(f"[{objective_type}] All random search efficiency values are above threshold {efficiency_threshold}. Random efficiency will not be plotted.")
            plot_random_mean_eff = pd.Series(dtype=float) # Empty series
            plot_random_std_eff = pd.Series(dtype=float)   # Empty series

        if not plot_random_mean_eff.empty:
            plt.plot(plot_random_mean_eff.index, plot_random_mean_eff, linestyle='--', label='Random Search Mean Efficiency', color='purple')
            plt.fill_between(plot_random_std_eff.index, 
                             plot_random_mean_eff - plot_random_std_eff, 
                             plot_random_mean_eff + plot_random_std_eff, 
                             alpha=0.2, color='purple', label='Random Search Efficiency Std Dev')
        else:
            # This case is covered by the print above, but good to be explicit.
            pass 

    plt.xlabel('Number of Function Evaluations (NFE)')
    plt.ylabel(f'Score per NFE ({score_type.capitalize()})')
    objective_str = f" ({objective_type} objective)" if objective_type else ""
    plt.title(f'Efficiency: {score_type.capitalize()} Score per NFE vs NFE by N Value{objective_str}')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Configuration')
    plt.tight_layout()
    return plt.gcf()

def create_n_m_heatmap(df: pd.DataFrame, score_type: str = 'final', objective_type: str = None):
    """
    Create a heatmap comparing different N/M configurations.
    
    Args:
        df: DataFrame containing the experiment results
        score_type: Either 'final' or 'overall' to specify which score to plot
        objective_type: Filter by objective type ('single', 'multi', or None for all)
    """
    if objective_type:
        filtered_df = df[df['objective_type'] == objective_type]
        if filtered_df.empty:
            print(f"No data for objective_type={objective_type}")
            return None
    else:
        filtered_df = df
    
    score_key = f'best_{score_type}_score'
    
    # Pivot the data to create a heatmap
    pivot_data = filtered_df.pivot_table(
        values=score_key,
        index='N',
        columns='M',
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        print(f"No data for heatmap with score_type={score_type}, objective_type={objective_type}")
        return None
    
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    ax = sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': f'Best {score_type.capitalize()} Score'}
    )
    
    # Add title
    objective_str = f" ({objective_type} objective)" if objective_type else ""
    plt.title(f'Average Best {score_type.capitalize()} Score by N/M Configuration{objective_str}')
    
    plt.tight_layout()
    return plt.gcf()

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SearchOverPaths and Random Search experiment results.")
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of base directories containing SearchOverPaths experiment results (e.g., examples/inference_scaling/inference_scaling_experiments/search_over_paths)."
    )
    parser.add_argument(
        "--random_search_base_dir",
        type=str,
        default=None, # Make it optional
        help="Base directory containing Random Search experiment results (e.g., examples/inference_scaling/inference_scaling_experiments/random)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save the generated plots and summary CSVs."
    )
    return parser.parse_args()

def main(base_dirs, random_search_base_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True) # Ensure output_dir exists early

    # --- Process Random Search Data First ---
    random_search_stats = None
    if random_search_base_dir:
        random_search_path = Path(random_search_base_dir)
        if random_search_path.exists() and random_search_path.is_dir():
            print(f"Processing random search results from: {random_search_path}")
            random_search_stats = process_random_search_data(str(random_search_path))
            if random_search_stats:
                print("Random search data processed successfully.")
            else:
                print("Failed to process random search data or no data found.")
        else:
            print(f"Random search directory not found or is not a directory: {random_search_base_dir}. Skipping random search plots.")
    else:
        print("Random search base directory not provided. Skipping random search plots.")

    # --- SearchOverPaths Analysis (Logic for passing random_search_stats added) ---
    df_sop_results = load_experiment_results(base_dirs)
    
    if df_sop_results.empty:
        print("No SearchOverPaths results loaded.")
    else:
        print(f"Loaded {len(df_sop_results)} SearchOverPaths experiment results.")
        
        sop_summary_path = Path(output_dir) / "sop_results_summary.csv"
        df_sop_results.to_csv(sop_summary_path, index=False)
        print(f"Saved combined SearchOverPaths results to {sop_summary_path}")

        sop_objective_types = df_sop_results['objective_type'].unique() # e.g., ['single', 'multi']
        print(f"Found SearchOverPaths objective types: {sop_objective_types}")

        for obj_type in sop_objective_types: # obj_type will be 'single' or 'multi'
            obj_df = df_sop_results[df_sop_results['objective_type'] == obj_type]
            
            # Determine the corresponding key for random_search_stats
            # SOP uses 'single'/'multi', random_search_stats uses 'single_objective'/'multi_objective'
            random_stats_key = f"{obj_type}_objective"
            current_random_stats = random_search_stats.get(random_stats_key, {}) if random_search_stats else {}

            # Extract specific random search data for this objective type
            rand_mean_score = current_random_stats.get("mean_cummax_score")
            rand_std_score = current_random_stats.get("std_cummax_score")
            rand_mean_eff = current_random_stats.get("mean_efficiency")
            rand_std_eff = current_random_stats.get("std_efficiency")
            
            for score_t in ['final', 'overall']: # For SOP scores
                # Score vs NFE plot by N value
                fig1 = create_score_vs_nfe_by_N_plot(
                    obj_df, 
                    score_type=score_t, 
                    objective_type=obj_type,
                    random_mean_scores=rand_mean_score,
                    random_std_scores=rand_std_score
                )
                if fig1:
                    fig1.savefig(Path(output_dir) / f"sop_score_vs_nfe_by_N_{score_t}_{obj_type}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig1)
                
                # Score vs NFE plot by M value
                fig2 = create_score_vs_nfe_by_M_plot(
                    obj_df, 
                    score_type=score_t, 
                    objective_type=obj_type,
                    random_mean_scores=rand_mean_score,
                    random_std_scores=rand_std_score
                )
                if fig2:
                    fig2.savefig(Path(output_dir) / f"sop_score_vs_nfe_by_M_{score_t}_{obj_type}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                
                # Efficiency line plot
                # SOP efficiency is calculated from score_t ('final' or 'overall'). 
                # Random search efficiency is based on its single 'combined_score' (effectively 'overall').
                fig3 = create_efficiency_line_plot(
                    obj_df, 
                    score_type=score_t, 
                    objective_type=obj_type,
                    random_mean_efficiency=rand_mean_eff,
                    random_std_efficiency=rand_std_eff
                )
                if fig3:
                    fig3.savefig(Path(output_dir) / f"sop_efficiency_line_{score_t}_{obj_type}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig3)
        
        print(f"All plots saved to {output_dir}")
        return random_search_stats

            
if __name__ == '__main__':
    args = parse_args()
    main(args.base_dirs, args.random_search_base_dir, args.output_dir) # Pass the new arg 