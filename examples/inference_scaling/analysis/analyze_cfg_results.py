"""
Analyze and visualize Classifier-Free Guidance (CFG) experiment results.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import re
import argparse
import glob
import colorsys # Added for HLS color manipulation
import matplotlib.colors as mcolors # Added for color conversion

# Configure seaborn for consistent plot aesthetics
sns.set_theme(style="whitegrid")

def load_pickle_data(file_path: str) -> Any:
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded pickle data from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None

def process_cfg_random_search_data(base_experiment_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Parses CFG random search CSV files from experiments structured like:
    cfg_random_w{weight}_nfe{NFE}_sa_target{sa_target}/all_molecules_log.csv
    Calculates cumulative maximum of 'combined_score'.
    Returns a dictionary where keys are experiment parameter tuples (weight, nfe_target, sa_target)
    and values are DataFrames with 'nfe', 'mean_cummax_score', 'std_cummax_score'.
    """
    all_processed_data: Dict[str, pd.DataFrame] = {}
    
    # Pattern to extract parameters from the directory name
    # Example: cfg_random_w3.0_nfe100_sa_target8.0
    # Example: cfg_random_w0.0_nfe500_sa_target8.0
    dir_pattern = re.compile(r"cfg_random_w([\d.]+)_nfe(\d+)_sa_target([\d.]+)")

    experiment_glob_pattern = os.path.join(base_experiment_dir, "cfg_random_w*_nfe*_sa_target*", "all_molecules_log.csv")
    file_paths = glob.glob(experiment_glob_pattern)

    if not file_paths:
        print(f"No CFG random search CSV files found matching pattern: {experiment_glob_pattern}")
        return all_processed_data

    # Group files by their parameters (weight, nfe_target, sa_target)
    # This is simplified as each unique directory is one "run" for a parameter set.
    # For CFG, we typically don't have multiple repeats for the *exact* same named experiment directory.
    # Each directory *is* a distinct experiment setting.

    grouped_experiments: Dict[Tuple[float, int, float], List[pd.DataFrame]] = {}

    for file_path in file_paths:
        path_obj = Path(file_path)
        experiment_dir_name = path_obj.parent.name # e.g., cfg_random_w3.0_nfe100_sa_target8.0
        
        match = dir_pattern.match(experiment_dir_name)
        if not match:
            print(f"Warning: Could not parse parameters from directory name: {experiment_dir_name}")
            continue
            
        cfg_weight = float(match.group(1))
        nfe_target = int(match.group(2)) # This is the NFE target for the experiment run
        sa_target = float(match.group(3))
        
        exp_params_key = (cfg_weight, nfe_target, sa_target)

        try:
            df = pd.read_csv(file_path)
            if 'iteration' not in df.columns or 'combined_score' not in df.columns:
                print(f"Warning: CSV {file_path} is missing 'iteration' or 'combined_score' column.")
                continue
            
            df = df.sort_values(by='iteration').reset_index(drop=True)
            # 'iteration' for random search is effectively the NFE.
            # Max NFE for this run is up to the nfe_target.
            df['nfe'] = df['iteration']
            df['cummax_score'] = df['combined_score'].cummax()
            
            # We expect one file per unique (cfg_w, nfe_target, sa_target) combination from cfg_sweep
            # If there were multiple repeats leading to multiple CSVs for the *same* params,
            # we'd need to average them. Here, each dir is a unique setting.
            # So, we store the single DataFrame of cummax_scores per experiment setting.
            
            # We will store the series directly, as we won't average over repeats here.
            # The "mean" and "std" will be trivial if there's only one run per param set.
            # For now, let's assume one run per (w, nfe_target, sa_target) config.
            # The structure will be: Dict[exp_key, DataFrame_with_cummax_scores_for_that_run]
            
            if exp_params_key not in grouped_experiments:
                grouped_experiments[exp_params_key] = []
            grouped_experiments[exp_params_key].append(df[['nfe', 'cummax_score']].set_index('nfe'))

        except pd.errors.EmptyDataError:
            print(f"Warning: CSV file {file_path} is empty.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Process grouped experiments to calculate mean and std if multiple runs existed (though typically not for CFG sweep per named dir)
    # For now, this loop will mostly just reformat.
    for params_key, df_list in grouped_experiments.items():
        if not df_list:
            continue

        # If there are multiple dataframes (e.g. from repeats not captured by unique dir names), concatenate and average
        if len(df_list) > 1:
            # Find the max NFE across these DFs to create a common index
            max_nfe_this_group = 0
            for series_df in df_list: # df_list contains DataFrames with 'cummax_score'
                if not series_df.index.empty: # series_df is indexed by 'nfe'
                    max_nfe_this_group = max(max_nfe_this_group, series_df.index.max())
            
            common_index = pd.Index(range(max_nfe_this_group + 1), name="nfe")
            
            aligned_scores = [
                s['cummax_score'].reindex(common_index).ffill().bfill() for s in df_list
            ]
            combined_scores_df = pd.concat(aligned_scores, axis=1)
            
            mean_scores = combined_scores_df.mean(axis=1)
            std_scores = combined_scores_df.std(axis=1)
        elif df_list: # Single DataFrame for this param set
            single_run_df = df_list[0] # This is a DataFrame with 'cummax_score' column, indexed by 'nfe'
            mean_scores = single_run_df['cummax_score']
            std_scores = pd.Series(0, index=single_run_df.index, dtype=float) # No std for a single run
        else:
            continue

        # Store as a DataFrame
        processed_df = pd.DataFrame({
            'mean_cummax_score': mean_scores,
            'std_cummax_score': std_scores
        })
        # Ensure 'nfe' is a column if it's only an index
        if processed_df.index.name == 'nfe':
            processed_df = processed_df.reset_index()

        all_processed_data[f"w{params_key[0]}_nfe{params_key[1]}_sa{params_key[2]}"] = processed_df
        print(f"Processed CFG Random Search data for: w={params_key[0]}, nfe_target={params_key[1]}, sa_target={params_key[2]}")

    return all_processed_data

def process_cfg_zero_order_data(base_experiment_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Parses CFG Zero-Order search CSV files from experiments structured like:
    cfg_zero_order_w{weight}_nfe{NFE}_sa_target{sa_target}/all_molecules_log.csv
    Skips experiments where NFE target was 500.
    Returns a dictionary where keys are experiment parameter tuples (weight, nfe_target, sa_target)
    and values are DataFrames with 'nfe', 'mean_score', 'std_score'.
    """
    all_processed_data: Dict[str, pd.DataFrame] = {}
    dir_pattern = re.compile(r"cfg_zero_order_w([\d.]+)_nfe(\d+)_sa_target([\d.]+)")

    experiment_glob_pattern = os.path.join(base_experiment_dir, "cfg_zero_order_w*_nfe*_sa_target*", "all_molecules_log.csv")
    file_paths = glob.glob(experiment_glob_pattern)

    if not file_paths:
        print(f"No CFG zero-order search CSV files found matching pattern: {experiment_glob_pattern}")
        return all_processed_data

    grouped_experiments: Dict[Tuple[float, int, float], List[pd.DataFrame]] = {}

    for file_path in file_paths:
        path_obj = Path(file_path)
        experiment_dir_name = path_obj.parent.name
        
        match = dir_pattern.match(experiment_dir_name)
        if not match:
            print(f"Warning: Could not parse parameters from directory name: {experiment_dir_name} for zero-order")
            continue
            
        cfg_weight = float(match.group(1))
        nfe_target = int(match.group(2))
        sa_target = float(match.group(3))

        # Skip NFE=500 for zero-order results
        if nfe_target == 500:
            print(f"Skipping zero-order experiment data for NFE target 500: {experiment_dir_name}")
            continue
        
        exp_params_key = (cfg_weight, nfe_target, sa_target)

        try:
            df = pd.read_csv(file_path)
            # Assuming 'iteration' is NFE and 'combined_score' is the score to plot directly
            if 'iteration' not in df.columns or 'combined_score' not in df.columns:
                print(f"Warning: CSV {file_path} is missing 'iteration' or 'combined_score' column.")
                continue
            
            df = df.sort_values(by='iteration').reset_index(drop=True)
            df['nfe'] = df['iteration'] # NFE is the iteration number
            # For zero-order, we use the score directly, not cumulative max
            df['score'] = df['combined_score']
            
            if exp_params_key not in grouped_experiments:
                grouped_experiments[exp_params_key] = []
            # Store the DataFrame with 'nfe' and 'score' columns, indexed by 'nfe'
            grouped_experiments[exp_params_key].append(df[['nfe', 'score']].set_index('nfe'))

        except pd.errors.EmptyDataError:
            print(f"Warning: CSV file {file_path} is empty.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    for params_key, df_list in grouped_experiments.items():
        if not df_list:
            continue

        if len(df_list) > 1:
            max_nfe_this_group = 0
            for series_df in df_list: # df_list contains DataFrames with 'score'
                if not series_df.index.empty:
                     max_nfe_this_group = max(max_nfe_this_group, series_df.index.max())
            
            common_index = pd.Index(range(max_nfe_this_group + 1), name="nfe")
            
            aligned_scores = [
                s['score'].reindex(common_index).ffill().bfill() for s in df_list
            ]
            combined_scores_df = pd.concat(aligned_scores, axis=1)
            
            mean_scores = combined_scores_df.mean(axis=1)
            std_scores = combined_scores_df.std(axis=1)
        elif df_list:
            single_run_df = df_list[0] # DataFrame with 'score' column, indexed by 'nfe'
            mean_scores = single_run_df['score']
            std_scores = pd.Series(0, index=single_run_df.index, dtype=float)
        else:
            continue

        processed_df = pd.DataFrame({
            'mean_score': mean_scores,
            'std_score': std_scores
        })
        if processed_df.index.name == 'nfe':
            processed_df = processed_df.reset_index()
            
        all_processed_data[f"w{params_key[0]}_nfe{params_key[1]}_sa{params_key[2]}"] = processed_df
        print(f"Processed CFG Zero-Order data for: w={params_key[0]}, nfe_target={params_key[1]}, sa_target={params_key[2]}")

    return all_processed_data

def plot_cfg_results_vs_baseline(
    cfg_data: Dict[str, pd.DataFrame], 
    baseline_random_data: Optional[Dict[str, Dict[str, pd.Series]]], 
    cfg_algorithm_type: str, # "Random Search (Cumulative Max)" or "Zero-Order Search (Raw Score)"
    baseline_objective_key: str, # e.g., "single_objective" or "multi_objective" from the pickle
    output_dir: str,
    plot_cumulative: bool # True for CFG Random, False for CFG Zero-Order
):
    """
    Plots CFG results against baseline random search data.
    Generates a plot for each unique SA target found in cfg_data.
    Each plot shows different CFG weights and NFE targets.
    """
    if not cfg_data:
        print(f"No CFG data provided for plotting {cfg_algorithm_type}.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract unique SA targets from CFG data keys (e.g., "w0.0_nfe100_sa8.0")
    sa_targets = sorted(list(set(float(key.split('_sa')[1]) for key in cfg_data.keys())))

    baseline_mean_series: Optional[pd.Series] = None
    baseline_std_series: Optional[pd.Series] = None

    if baseline_random_data and baseline_objective_key in baseline_random_data:
        baseline_mean_series = baseline_random_data[baseline_objective_key].get("mean_cummax_score")
        baseline_std_series = baseline_random_data[baseline_objective_key].get("std_cummax_score")
        if baseline_mean_series is None:
            print(f"Warning: 'mean_cummax_score' not found in baseline data for key '{baseline_objective_key}'. Baseline will not be plotted.")
    elif baseline_random_data:
        print(f"Warning: Objective key '{baseline_objective_key}' not found in baseline data. Baseline will not be plotted.")
    else:
        print("Warning: Baseline random data not provided. Plotting CFG results only.")

    for sa_target_val in sa_targets:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        # Plot baseline random search first
        if baseline_mean_series is not None:
            ax.plot(baseline_mean_series.index, baseline_mean_series, label=f'Baseline Random ({baseline_objective_key} cummax)', color='black', linestyle='--', zorder=1)
            if baseline_std_series is not None:
                ax.fill_between(baseline_std_series.index, 
                                baseline_mean_series - baseline_std_series, 
                                baseline_mean_series + baseline_std_series, 
                                color='black', alpha=0.1, zorder=1)
        
        # Plot CFG data for the current SA target
        # Group by weight, then plot lines for different NFEs for that weight
        experiments_for_sa_target = {
            k: v for k, v in cfg_data.items() if float(k.split('_sa')[1]) == sa_target_val
        }
        
        # Extract weights and nfe_targets for legend and coloring
        # Example key: "w0.0_nfe100_sa8.0"
        unique_weights = sorted(list(set(float(key.split('_nfe')[0].split('w')[1]) for key in experiments_for_sa_target.keys())))
        
        num_weights = len(unique_weights)
        # Get a color palette for weights
        weight_colors = sns.color_palette("viridis", n_colors=num_weights if num_weights > 0 else 1)
        
        line_styles = ['-', ':', '-.', '--'] # For different NFE targets under the same weight

        for i, weight_val in enumerate(unique_weights):
            base_color_for_weight_from_palette = weight_colors[i % len(weight_colors)] # Main color for this weight
            
            # Adjust the base color to be slightly lighter for the start of the NFE sequence
            try:
                rgb_tuple = mcolors.to_rgb(base_color_for_weight_from_palette)
                h, l, s = colorsys.rgb_to_hls(*rgb_tuple)
                # Increase lightness: add 0.15, capping at 1.0 to avoid going beyond white
                # and ensuring it's not less than current l if l is already high.
                new_l = min(1.0, l + 0.15) 
                # To prevent overly bright colors from becoming dull, ensure saturation isn't too low
                # new_s = max(0.3, s) # Optional: ensure minimum saturation
                adjusted_start_color_rgb = colorsys.hls_to_rgb(h, new_l, s) # Use original saturation s
            except Exception as e:
                print(f"Warning: Could not adjust color {base_color_for_weight_from_palette}, using original. Error: {e}")
                adjusted_start_color_rgb = base_color_for_weight_from_palette


            weight_specific_experiments = {
                k: v for k, v in experiments_for_sa_target.items() 
                if float(k.split('_nfe')[0].split('w')[1]) == weight_val
            }
            
            # Sort by NFE target for consistent line style and color palette application
            sorted_nfe_exp_keys = sorted(weight_specific_experiments.keys(), key=lambda k: int(k.split('_nfe')[1].split('_sa')[0]))

            num_nfe_lines_for_this_weight = len(sorted_nfe_exp_keys)

            if num_nfe_lines_for_this_weight > 0:
                # Create a dark palette starting from the adjusted (lighter) base color
                nfe_specific_color_palette = sns.dark_palette(
                    adjusted_start_color_rgb, # Use the HLS-adjusted lighter color as the start
                    n_colors=num_nfe_lines_for_this_weight,
                    reverse=False # Starts with adjusted_start_color_rgb and gets darker
                )
            else: 
                nfe_specific_color_palette = [adjusted_start_color_rgb] 

            for j, exp_key in enumerate(sorted_nfe_exp_keys):
                df_exp = experiments_for_sa_target[exp_key]
                nfe_target_val = int(exp_key.split('_nfe')[1].split('_sa')[0])
                
                label = f'CFG w={weight_val}, nfe_target={nfe_target_val}'
                # Assign color from the NFE-specific palette for this weight
                if num_nfe_lines_for_this_weight > 0:
                    color = nfe_specific_color_palette[j]
                else:
                    color = base_color_for_weight_from_palette # Fallback
                
                style = line_styles[j % len(line_styles)]
                
                score_col = 'mean_cummax_score' if plot_cumulative else 'mean_score'
                std_col = 'std_cummax_score' if plot_cumulative else 'std_score'

                if 'nfe' in df_exp.columns and score_col in df_exp.columns:
                    ax.plot(df_exp['nfe'], df_exp[score_col], label=label, color=color, linestyle=style, marker='.', zorder=2)
                    if std_col in df_exp.columns and df_exp[std_col].sum() > 0: # Check if std is not all zeros
                        ax.fill_between(df_exp['nfe'], 
                                        df_exp[score_col] - df_exp[std_col], 
                                        df_exp[score_col] + df_exp[std_col], 
                                        color=color, alpha=0.15, zorder=2)
                else:
                    print(f"Warning: 'nfe' or '{score_col}' column missing in processed data for {exp_key}. Skipping plot for this entry.")

        ax.set_xlabel("Number of Function Evaluations (NFE)")
        ax.set_ylabel("Score")
        ax.set_title(f"CFG {cfg_algorithm_type} vs. Baseline Random (SA Target: {sa_target_val})")
        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        
        plot_filename = f"cfg_{cfg_algorithm_type.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-','_')}_vs_baseline_sa_target_{sa_target_val}.png"
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
        print(f"Saved plot: {os.path.join(output_dir, plot_filename)}")
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CFG experiment results and compare with baseline random search.")
    parser.add_argument(
        "--cfg_base_dir",
        type=str,
        required=True,
        help="Base directory containing CFG experiment subdirectories (e.g., inference_scaling_experiments/cfg_sweep2)."
    )
    parser.add_argument(
        "--baseline_pickle_path",
        type=str,
        required=True,
        help="Path to the baseline random_search_data.pkl file (e.g., examples/inference_scaling/sop_plots/random_search_data.pkl)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cfg_analysis_results",
        help="Directory to save the generated plots."
    )
    parser.add_argument(
        "--baseline_objective_key",
        type=str,
        default="single_objective", # Or make required if it varies often
        help="Objective key from the baseline pickle to use for comparison (e.g., 'single_objective', 'multi_objective')."
    )
    return parser.parse_args()

def main(cfg_base_dir: str, baseline_pickle_path: str, output_dir: str, baseline_objective_key: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # Load baseline random search data
    print(f"Loading baseline random search data from: {baseline_pickle_path}")
    baseline_data = load_pickle_data(baseline_pickle_path)
    if baseline_data is None:
        print("Failed to load baseline data. Plots will not include baseline comparison.")
    
    # Process CFG Random Search data
    print(f"Processing CFG Random Search data from: {cfg_base_dir}")
    cfg_random_processed_data = process_cfg_random_search_data(cfg_base_dir)
    if cfg_random_processed_data:
        plot_cfg_results_vs_baseline(
            cfg_data=cfg_random_processed_data,
            baseline_random_data=baseline_data,
            cfg_algorithm_type="Random Search (Cumulative Max)",
            baseline_objective_key=baseline_objective_key,
            output_dir=output_dir,
            plot_cumulative=True
        )
    else:
        print("No CFG Random Search data processed.")

    # Process CFG Zero-Order Search data
    print(f"\nProcessing CFG Zero-Order Search data from: {cfg_base_dir}")
    cfg_zero_order_processed_data = process_cfg_zero_order_data(cfg_base_dir)
    if cfg_zero_order_processed_data:
        plot_cfg_results_vs_baseline(
            cfg_data=cfg_zero_order_processed_data,
            baseline_random_data=baseline_data,
            cfg_algorithm_type="Zero-Order Search (Raw Score)",
            baseline_objective_key=baseline_objective_key,
            output_dir=output_dir,
            plot_cumulative=False
        )
    else:
        print("No CFG Zero-Order Search data processed.")
    
    print(f"\nCFG analysis complete. Plots saved in {output_dir}")

if __name__ == '__main__':
    args = parse_args()
    main(
        cfg_base_dir=args.cfg_base_dir, 
        baseline_pickle_path=args.baseline_pickle_path, 
        output_dir=args.output_dir, 
        baseline_objective_key=args.baseline_objective_key
    )
    # This is a placeholder for testing the functions as they are developed.
    # Actual argument parsing and main logic will be added later.
    # print("analyze_cfg_results.py script started.")

    # Example usage (assuming you have some data in the expected structure)
    # cfg_random_data_dir = "inference_scaling_experiments/cfg_sweep2" # Replace with your actual path
    # processed_cfg_random = process_cfg_random_search_data(cfg_random_data_dir)
    
    # for key, df in processed_cfg_random.items():
    #     print(f"\\nResults for {key}:")
    #     print(df.head())

    # baseline_random_pickle = "examples/inference_scaling/sop_plots/random_search_data.pkl" # Replace with your actual path
    # baseline_random_data = load_pickle_data(baseline_random_pickle)
    # if baseline_random_data:
    #    print("\\nBaseline Random Search Data (example from one objective type):")
    #    if "single_objective" in baseline_random_data:
    #        print(baseline_random_data["single_objective"]["mean_cummax_score"].head())
    # pass 