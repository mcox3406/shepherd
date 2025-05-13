"""
Run inference scaling experiments for ShEPhERD.

This script allows running individual search algorithms with configurable parameters
for hyperparameter sweeps and longer experiments.
"""

import os
import sys
from copy import deepcopy
import torch
import numpy as np
import argparse
import pickle
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import logging
from rdkit import Chem

from shepherd.lightning_module import LightningModule
from shepherd.inference_scaling import (
    ShepherdModelRunner,
    SAScoreVerifier,
    CLogPVerifier,
    MultiObjectiveVerifier,
    RandomSearch,
    ZeroOrderSearch,
    GuidedSearch,
    SearchOverPaths,
    create_rdkit_molecule,
    get_xyz_content
)


# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference scaling experiment with ShEPhERD')
    
    # required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--algorithm', type=str, required=True, choices=['random', 'zero_order', 'guided', 'search_over_paths'],
                        help='Search algorithm to use (random, zero_order, guided, search_over_paths)')
    
    # model configuration
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cpu, cuda, mps). Default: auto-select best device')
    parser.add_argument('--n_atoms', type=int, default=25,
                        help='Number of atoms to generate')
    parser.add_argument('--n_pharm', type=int, default=5,
                        help='Number of pharmacophores')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    
    # repeat and experiment ID options
    parser.add_argument('--repeat_N', type=int, default=1,
                        help='Number of times to repeat the experiment')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Optional ID to append to the random experiment name')
    
    # search parameters
    parser.add_argument('--num_trials', type=int, default=50,
                        help='Number of trials for random search')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of steps for zero-order search')
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help='Number of neighbors to evaluate per step for zero-order search')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Step size for zero-order search perturbation')
    
    # guided search parameters
    parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size for guided search')
    parser.add_argument('--num_generations', type=int, default=10,
                        help='Number of generations for guided search')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                        help='Mutation rate for guided search')
    parser.add_argument('--elite_fraction', type=float, default=0.1,
                        help='Fraction of elite individuals for guided search')
    
    # verifier parameters
    parser.add_argument('--sa_weight', type=float, default=1.0,
                        help='Weight for synthetic accessibility score')
    parser.add_argument('--clogp_weight', type=float, default=0.0,
                        help='Weight for cLogP score')
    
    # output control
    parser.add_argument('--output_dir', type=str, default='inference_scaling_experiments',
                        help='Base directory to save experiment results')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated from parameters)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after the experiment')
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='Save checkpoints during search for later resumption')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Interval for saving checkpoints (iterations)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    
    # sampler configuration
    parser.add_argument('--sampler_type', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='Sampling algorithm (ddpm or ddim)')
    parser.add_argument('--num_sampling_steps', type=int, default=None,
                        help='Number of steps for the sampler (defaults to T for DDPM, T for DDIM if None)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                        help='Eta parameter for DDIM sampling (0.0 for deterministic)')
    
    # search over paths parameters
    parser.add_argument('--num_paths_N', type=int, default=10,
                        help='Number of paths to search over')
    parser.add_argument('--path_width_M', type=int, default=5,
                        help='Number of samples for each path')
    parser.add_argument('--initial_t_idx', type=float, default=355,
                        help='Initial time to denoise to')
    parser.add_argument('--delta_f', type=float, default=312,
                        help='Number of steps to forward noise')
    parser.add_argument('--delta_b', type=float, default=324,
                        help='Number of steps to reverse noise. Must be larger than delta_f.')

    # classifier-free guidance (CFG) parameters
    parser.add_argument('--do_property_cfg', action='store_true', default=False,
                        help='Enable Classifier-Free Guidance (CFG) mode')
    parser.add_argument('--cfg_weight', type=float, default=1.0,
                        help='Scale for Classifier-Free Guidance')
    parser.add_argument('--sa_score_target_value', type=float, default=1.0,
                        help='Target value for the guided property in CFG')
    
    return parser.parse_args()


def get_device(device_arg):
    """Select the best available device"""
    if device_arg is not None:
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS is available, but using CPU for better compatibility")
        return torch.device('cpu')
    else:
        return torch.device('cpu')


def generate_experiment_name(args):
    """Generate an experiment name based on arguments"""
    if args.exp_name:
        return args.exp_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    name_parts = []
    
    if args.algorithm == 'random':
        name_parts.append(f"{args.algorithm}_trials{args.num_trials}")
        if args.run_id:
            name_parts.append(f"run{args.run_id}")
    elif args.algorithm == 'zero_order':
        name_parts.append(f"{args.algorithm}_steps{args.num_steps}_nbrs{args.num_neighbors}_ss{args.step_size}")
    elif args.algorithm == 'guided':
        name_parts.append(f"{args.algorithm}_pop{args.pop_size}_gen{args.num_generations}_mut{args.mutation_rate}_elite{args.elite_fraction}")
    elif args.algorithm == 'search_over_paths':
        name_parts.append(f"{args.algorithm}_paths{args.num_paths}_len{args.path_length}_ss{args.path_step_size}")
    
    # add sampler info if not default DDPM
    if args.sampler_type != 'ddpm':
        name_parts.append(f"sampler_{args.sampler_type}")
        if args.num_sampling_steps is not None:
            name_parts.append(f"steps{args.num_sampling_steps}")
        if args.sampler_type == 'ddim':
             name_parts.append(f"eta{args.ddim_eta}")

    # add CFG info if enabled
    if args.do_property_cfg:
        name_parts.append("cfg")
        name_parts.append(f"cfg_weight{args.cfg_weight}")
        name_parts.append(f"sa_score_target{args.sa_score_target_value}")

    name_parts.append(timestamp)
    return "_".join(name_parts)


def save_results(results, output_path):
    """Save experiment results to the specified output path"""
    try:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"Saving results to: {output_path}")
        
        pickle_path = output_path / "results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Full results saved to: {pickle_path}")
        
        # save summary and configuration as JSON
        summary = {
            'algorithm': results['algorithm'],
            'best_score': results.get('best_score', 0),
            'duration_seconds': results.get('duration_seconds', 0),
            'num_evaluations': len(results.get('scores', [])),
            'nfe': results.get('metadata', {}).get('nfe', 0),
            'best_sa_score': results.get('best_sa_score'),
            'best_clogp_score': results.get('best_clogp_score'),
            'config': results.get('config', {}),
            'best_smiles': results.get('best_smiles', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add SearchOverPaths specific fields if they exist
        if 'metadata' in results and isinstance(results['metadata'], dict):
            if 'best_final_score' in results['metadata']:
                summary['best_final_score'] = results['metadata']['best_final_score']
            if 'best_final_path_idx' in results['metadata']:
                summary['best_final_path_idx'] = results['metadata']['best_final_path_idx']
            if 'best_overall_path_idx' in results['metadata']:
                summary['best_overall_path_idx'] = results['metadata']['best_overall_path_idx']
        
        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        print(f"Summary saved to: {summary_path}")
        
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def plot_results(results, output_path):
    """Generate plots for experiment results"""
    scores = results.get('scores', [])
    
    if not scores:
        print("No scores to plot")
        return
    
    # score progression plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, 'b-o')
    
    # highlight best score
    best_idx = np.argmax(scores)
    plt.scatter(best_idx + 1, scores[best_idx], color='r', marker='*', s=200,
                label=f'Best Score: {scores[best_idx]:.4f}')
    
    # add baseline if available
    if 'baseline_score' in results:
        plt.axhline(y=results['baseline_score'], color='k', linestyle='--',
                    label=f'Baseline: {results["baseline_score"]:.4f}')
    
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Score')
    plt.title(f"{results['algorithm']} Search Performance")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(output_path / "score_progression.png")
    print(f"Plot saved to {output_path / 'score_progression.png'}")


def run_experiment(args):
    """Run the inference scaling experiment with the specified parameters"""
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # generate experiment name and set up output directory
    exp_name = generate_experiment_name(args)
    output_path = Path(args.output_dir) / exp_name
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Experiment output directory: {output_path}")
    
    # create subdirectory for all molecules
    all_mols_dir = output_path / "all_molecules"
    all_mols_dir.mkdir(exist_ok=True)
    all_mols_log_path = output_path / "all_molecules_log.csv"

    # initialize CSV log file
    log_header = ['filename', 'algorithm', 'iteration', 'sub_iteration', 'is_initial', 'is_elite',
                  'smiles', 'sa_score', 'clogp_score', 'combined_score']
    with open(all_mols_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(log_header)

    # save configuration
    config = vars(args)
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)
    
    print(f"Running experiment: {exp_name}")
    print(f"Loading model from {args.checkpoint}")
    
    try:
        # load model
        model_pl = LightningModule.load_from_checkpoint(args.checkpoint, weights_only=True)
        model_pl.eval()
        model_pl.to(device)
        model_pl.model.device = device
        print("Model loaded successfully")
        
        # set up inference parameters (COPYING FROM INTEGRATION TEST)
        inject_noise_at_ts = list(np.arange(130, 80, -1))
        inject_noise_scales = [1.0] * len(inject_noise_at_ts)
        harmonize = True
        harmonize_ts = [80]
        harmonize_jumps = [20]
        
        print(f"Creating model runner for {args.n_atoms} atoms and {args.n_pharm} pharmacophores")
        model_runner = ShepherdModelRunner(
            model_pl=model_pl,
            batch_size=args.batch_size,
            N_x1=args.n_atoms,
            N_x4=args.n_pharm,
            unconditional=True,
            device=str(device),
            save_all_noise=True,
            prior_noise_scale=1.0,
            denoising_noise_scale=1.0,
            inject_noise_at_ts=inject_noise_at_ts,
            inject_noise_scales=inject_noise_scales,
            harmonize=harmonize,
            harmonize_ts=harmonize_ts,
            harmonize_jumps=harmonize_jumps,
            sampler_type=args.sampler_type,
            num_steps=args.num_sampling_steps,
            ddim_eta=args.ddim_eta,
            do_property_cfg=args.do_property_cfg,
            cfg_weight=args.cfg_weight,
            sa_score=args.sa_score_target_value
        )
        
        print("Creating verifiers")
        sa_verifier = SAScoreVerifier(weight=args.sa_weight)
        clogp_verifier = CLogPVerifier(weight=args.clogp_weight)
        multi_verifier = MultiObjectiveVerifier([sa_verifier, clogp_verifier])
        
        def comprehensive_save_callback(**kwargs):
            try:
                sample = kwargs['sample']
                combined_score = kwargs['score']
                algorithm = kwargs['algorithm']
                iteration = kwargs['iteration']
                sub_iteration = kwargs.get('sub_iteration', -1) # default for random
                is_initial = kwargs.get('is_initial', False) # default for non-guided
                is_elite = kwargs.get('is_elite', False) # default for non-guided

                # 1. generate filename
                prefix = f"{algorithm}_i{iteration:04d}"
                if sub_iteration != -1:
                    prefix += f"_s{sub_iteration:04d}"
                filename_xyz = f"{prefix}.xyz"
                filepath_xyz = all_mols_dir / filename_xyz

                # 2. generate and save xyz content
                xyz_content = get_xyz_content(deepcopy(sample))
                if xyz_content:
                    with open(filepath_xyz, 'w') as f:
                        f.write(xyz_content)
                else:
                    filename_xyz = "error_generating_xyz"

                # 3. calculate individual properties and smiles
                sa_score = sa_verifier(sample)
                clogp_score = clogp_verifier(sample)
                smiles = "error_creating_mol"
                try:
                    mol = create_rdkit_molecule(sample)
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                except Exception as e:
                    logging.debug(f"Could not get SMILES for {filename_xyz}: {e}")
                    smiles = "error_generating_smiles"

                # 4. append to csv log
                log_row = [filename_xyz, algorithm, iteration, sub_iteration, is_initial, is_elite,
                           smiles, sa_score, clogp_score, combined_score]
                with open(all_mols_log_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(log_row)

            except Exception as e:
                logging.error(f"Error in comprehensive_save_callback: {e}", exc_info=True)

        # generate baseline sample for comparison
        print("generating baseline sample")
        baseline_sample = model_runner()
        baseline_score = multi_verifier(baseline_sample)
        print(f"baseline score: {baseline_score:.4f}")
        
        # initialize results dictionary
        results = {
            'algorithm': args.algorithm,
            'baseline_sample': baseline_sample,
            'baseline_score': baseline_score,
            'config': config,
            'start_time': time.time()
        }
        
        # run the selected search algorithm
        if args.algorithm == 'random':
            print(f"running random search with {args.num_trials} trials")
            search = RandomSearch(multi_verifier, model_runner)
            
            # run search
            best_sample, best_score, scores, metadata = search.search(
                num_trials=args.num_trials,
                verbose=args.verbose,
                device=str(device),
                callback=comprehensive_save_callback
            )
        
        elif args.algorithm == 'zero_order':
            print(f"Running Zero-Order Search with {args.num_steps} steps, {args.num_neighbors} neighbors")
            search = ZeroOrderSearch(multi_verifier, model_runner)
            
            # run search
            best_sample, best_score, scores, metadata = search.search(
                num_steps=args.num_steps,
                num_neighbors=args.num_neighbors,
                step_size=args.step_size,
                verbose=args.verbose,
                device=str(device),
                callback=comprehensive_save_callback
            )
        
        elif args.algorithm == 'guided':
            print(f"Running Guided Search with {args.pop_size} population, {args.num_generations} generations")
            search = GuidedSearch(multi_verifier, model_runner)

            # run search
            best_sample, best_score, scores, metadata = search.search(
                pop_size=args.pop_size,
                num_generations=args.num_generations,
                mutation_rate=args.mutation_rate,
                elite_fraction=args.elite_fraction,
                verbose=args.verbose,
                device=str(device),
                callback=comprehensive_save_callback
            )
        
        elif args.algorithm == 'search_over_paths':
            print(f"Running Search Over Paths with {args.num_paths_N} paths and {args.path_width_M} samples per path")
            search = SearchOverPaths(multi_verifier, model_runner)
            
            # Custom callback for SearchOverPaths that adapts to its different structure
            def search_over_paths_callback(**kwargs):
                try:
                    sample = kwargs['best_structure_output']
                    score = kwargs['best_score']
                    algorithm = kwargs['algorithm']
                    iteration = kwargs['iteration']
                    sub_iteration = kwargs['sub_iteration']
                    best_rdmol = kwargs.get('best_rdmol')
                    scores = kwargs.get('scores', [])

                    # 1. generate filename
                    prefix = f"{algorithm}_i{iteration:04d}_s{sub_iteration:04d}"
                    filename_xyz = f"{prefix}.xyz"
                    filepath_xyz = all_mols_dir / filename_xyz

                    # 2. generate and save xyz content
                    xyz_content = get_xyz_content(sample)
                    if xyz_content:
                        with open(filepath_xyz, 'w') as f:
                            f.write(xyz_content)
                    else:
                        filename_xyz = "error_generating_xyz"

                    # 3. calculate individual properties and smiles
                    sa_score = sa_verifier(sample)
                    clogp_score = clogp_verifier(sample)
                    smiles = "error_creating_mol"
                    if best_rdmol:
                        try:
                            smiles = Chem.MolToSmiles(best_rdmol)
                        except Exception as e:
                            logging.debug(f"Could not get SMILES for {filename_xyz}: {e}")
                            smiles = "error_generating_smiles"

                    # 4. append to csv log
                    log_row = [filename_xyz, algorithm, iteration, sub_iteration, False, False,
                               smiles, sa_score, clogp_score, score]
                    with open(all_mols_log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(log_row)

                except Exception as e:
                    logging.error(f"Error in search_over_paths_callback: {e}", exc_info=True)
            
            # run search
            (best_final_score, best_overall_score, best_final_path_idx, best_overall_path_idx,
             best_final_sample, best_overall_sample, best_final_rdmol, best_overall_rdmol,
             top_N, all_scores_history, metadata) = search.search(
                N_paths=args.num_paths_N,
                M_expansion_per_path=args.path_width_M,
                sigma_t_initial=args.initial_t_idx,
                delta_f_steps=args.delta_f,
                delta_b_steps=args.delta_b,
                N_x1=args.n_atoms,
                N_x4=args.n_pharm,
                verbose=args.verbose,
                device=str(device),
                callback=search_over_paths_callback
            )
            
            # Adapt the return values to match the expected format
            best_sample = best_overall_sample  # Use the best overall sample
            best_score = best_overall_score    # Use the best overall score
            scores = [d['best_overall_score'] for d in list(top_N.values())]  # Use best scores from each path
            
            # Add additional metadata specific to SearchOverPaths
            metadata.update({
                'best_final_score': best_final_score,
                'best_final_path_idx': best_final_path_idx,
                'best_overall_path_idx': best_overall_path_idx,
                'all_scores_history': all_scores_history
            })
        
        # calculate experiment duration
        end_time = time.time()
        duration = end_time - results['start_time']
        
        # analyze the best molecule
        if best_sample is not None:
            mol = create_rdkit_molecule(best_sample)
            if mol is not None:
                from rdkit import Chem
                smiles = Chem.MolToSmiles(mol)
                print(f"Best molecule SMILES: {smiles}")
                results['best_smiles'] = smiles
        
        # update results
        results.update({
            'best_sample': best_sample,
            'best_score': best_score,
            'scores': scores,
            'metadata': metadata,
            'duration_seconds': duration,
            'end_time': end_time
        })
        
        # print summary
        print("\nExperiment Summary:")
        print(f"Algorithm: {args.algorithm}")
        print(f"Best Score: {best_score:.4f}")
        if baseline_score == 0:
            print(f"Improvement over Baseline: {best_score - baseline_score:.4f} (N/A%)")
        else:
            print(f"Improvement over Baseline: {best_score - baseline_score:.4f} ({(best_score - baseline_score) / baseline_score * 100:.2f}%)")
        print(f"Duration: {duration:.2f} seconds")

        nfe = metadata.get('nfe', 0)
        print(f"Total Evaluations (NFE): {nfe}")

        best_sa_score = None
        best_clogp_score = None
        if best_sample is not None:
            try:
                best_sa_score = sa_verifier(best_sample)
                best_clogp_score = clogp_verifier(best_sample)
                print(f"Best Individual Scores - SA: {best_sa_score:.4f}, cLogP: {best_clogp_score:.4f}")
            except Exception as e:
                logging.warning(f"Could not calculate individual scores for best sample: {e}")

        # update results dictionary with NFE and individual scores
        results.update({
            'nfe': nfe,
            'best_sa_score': best_sa_score,
            'best_clogp_score': best_clogp_score
        })

        # Save results
        print("\nSaving results...")
        if save_results(results, output_path):
            print(f"Results saved successfully to {output_path}")
        else:
            print("Warning: Failed to save some results")
        
        # Generate plots if requested
        if args.plot:
            print("\nGenerating plots...")
            plot_results(results, output_path)
        
        return results
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        
        # try to save partial results if we have any
        try:
            partial_results = {
                'algorithm': args.algorithm,
                'error': str(e),
                'config': config,
                'traceback': traceback.format_exc()
            }
            with open(output_path / "error_results.pkl", 'wb') as f:
                pickle.dump(partial_results, f)
            print(f"Partial results saved to {output_path}")
        except:
            pass
        
        return None


if __name__ == "__main__":
    args = parse_args()
    
    original_output_dir = args.output_dir 
    original_exp_name = args.exp_name # Store original exp_name if provided

    if args.repeat_N > 1:
        print(f"--- Experiment configured to run {args.repeat_N} times ---")

    all_run_results_summary = []

    for i in range(args.repeat_N):
        print(f"--- Starting run {i+1}/{args.repeat_N} ---")
        
        current_args = argparse.Namespace(**vars(args)) # Create a mutable copy for this iteration

        # Modify output directory for this specific run to prevent overwrites
        # Results for repeat i will go into original_output_dir / repeat_i / <generated_exp_name>
        if args.run_id is not None:
            current_args.output_dir = str(Path(original_output_dir) / f"{args.run_id}" / f"repeat_{i}")
        else:
            current_args.output_dir = str(Path(original_output_dir) / f"repeat_{i}")
        
        # If an original exp_name was given, append repeat index to it for clarity in subfolder naming.
        # Otherwise, generate_experiment_name will create a unique name (potentially with run_id).
        if original_exp_name:
            current_args.exp_name = f"{original_exp_name}_repeat{i}"
        else:
            # Ensure exp_name is None so generate_experiment_name creates a new one based on params + run_id
            current_args.exp_name = None 

        # The generate_experiment_name function will be called inside run_experiment,
        # using current_args.exp_name (if set) or generating one (using current_args.run_id for random).
        
        results = run_experiment(current_args) 
        
        if results: # Store key summary metrics if run succeeded
            all_run_results_summary.append({
                'run_index': i,
                'best_score': results.get('best_score'),
                'nfe': results.get('metadata', {}).get('nfe'),
                'output_path': results.get('final_output_path') # Get the path where results were saved
            })
        else:
            print(f"--- Run {i+1}/{args.repeat_N} failed or did not return results ---")

    print(f"--- Completed {args.repeat_N} runs ---")

    if args.repeat_N > 1 and all_run_results_summary:
        print("\n--- Summary of all runs ---")
        for res_summary in all_run_results_summary:
            # Ensure values exist before formatting
            score_str = f"{res_summary.get('best_score', 'N/A'):.4f}" if isinstance(res_summary.get('best_score'), (int, float)) else str(res_summary.get('best_score', 'N/A'))
            nfe_str = str(res_summary.get('nfe', 'N/A'))
            path_str = str(res_summary.get('output_path', 'N/A'))
            print(f"Run {res_summary.get('run_index', '?')}: Best Score: {score_str}, NFE: {nfe_str}, Output Dir: {path_str}")
        
        # Example: Calculate and print mean/std of best scores
        valid_scores = [r['best_score'] for r in all_run_results_summary if isinstance(r.get('best_score'), (int, float))]
        if valid_scores:
            print(f"\nMean Best Score across {len(valid_scores)} successful runs: {np.mean(valid_scores):.4f}")
            print(f"Std Dev Best Score across {len(valid_scores)} successful runs: {np.std(valid_scores):.4f}")
