"""
Run inference scaling experiments for ShEPhERD.

This script allows running individual search algorithms with configurable parameters
for hyperparameter sweeps and longer experiments.
"""

import os
import sys
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
    QEDVerifier,
    MultiObjectiveVerifier,
    RandomSearch,
    ZeroOrderSearch,
    GuidedSearch,
    create_rdkit_molecule,
    get_xyz_content
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference scaling experiment with ShEPhERD')
    
    # required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--algorithm', type=str, required=True, choices=['random', 'zero_order', 'guided'],
                        help='Search algorithm to use (random, zero_order, guided)')
    
    # model configuration
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cpu, cuda, mps). Default: auto-select best device')
    parser.add_argument('--n_atoms', type=int, default=25,
                        help='Number of atoms to generate')
    parser.add_argument('--n_pharm', type=int, default=5,
                        help='Number of pharmacophores')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    
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
    parser.add_argument('--qed_weight', type=float, default=0.0,
                        help='Weight for QED (drug-likeness) score')
    
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
    elif args.algorithm == 'zero_order':
        name_parts.append(f"{args.algorithm}_steps{args.num_steps}_nbrs{args.num_neighbors}_ss{args.step_size}")
    elif args.algorithm == 'guided':
        name_parts.append(f"{args.algorithm}_pop{args.pop_size}_gen{args.num_generations}_mut{args.mutation_rate}_elite{args.elite_fraction}")
    
    # add sampler info if not default DDPM
    if args.sampler_type != 'ddpm':
        name_parts.append(f"sampler_{args.sampler_type}")
        if args.num_sampling_steps is not None:
            name_parts.append(f"steps{args.num_sampling_steps}")
        if args.sampler_type == 'ddim':
             name_parts.append(f"eta{args.ddim_eta}")

    name_parts.append(timestamp)
    return "_".join(name_parts)


def save_results(results, output_path):
    """Save experiment results to the specified output path"""
    output_path.mkdir(exist_ok=True, parents=True)
    
    # save all results in pickle format
    with open(output_path / "results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # save summary and configuration as JSON
    summary = {
        'algorithm': results['algorithm'],
        'best_score': results.get('best_score', 0),
        'duration_seconds': results.get('duration_seconds', 0),
        'num_evaluations': len(results.get('scores', [])),
        'nfe': results.get('nfe', 0),
        'best_sa_score': results.get('best_sa_score'),
        'best_clogp_score': results.get('best_clogp_score'),
        'best_qed_score': results.get('best_qed_score'),
        'config': results.get('config', {})
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_path}")


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
    
    # create subdirectory for all molecules
    all_mols_dir = output_path / "all_molecules"
    all_mols_dir.mkdir(exist_ok=True)
    all_mols_log_path = output_path / "all_molecules_log.csv"

    # initialize CSV log file
    log_header = ['filename', 'algorithm', 'iteration', 'sub_iteration', 'is_initial', 'is_elite',
                  'smiles', 'sa_score', 'clogp_score', 'qed_score', 'combined_score']
    with open(all_mols_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(log_header)

    # save configuration
    config = vars(args)
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
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
            ddim_eta=args.ddim_eta
        )
        
        print("Creating verifiers")
        sa_verifier = SAScoreVerifier(weight=args.sa_weight)
        clogp_verifier = CLogPVerifier(weight=args.clogp_weight)
        qed_verifier = QEDVerifier(weight=args.qed_weight)
        multi_verifier = MultiObjectiveVerifier([sa_verifier, clogp_verifier, qed_verifier])
        
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
                xyz_content = get_xyz_content(sample)
                if xyz_content:
                    with open(filepath_xyz, 'w') as f:
                        f.write(xyz_content)
                else:
                    filename_xyz = "error_generating_xyz"

                # 3. calculate individual properties and smiles
                sa_score = sa_verifier(sample)
                clogp_score = clogp_verifier(sample)
                qed_score = qed_verifier(sample)
                smiles = "error_creating_mol"
                try:
                    mol = create_rdkit_molecule(sample)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                except Exception as e:
                    logging.debug(f"Could not get SMILES for {filename_xyz}: {e}")
                    smiles = "error_generating_smiles"

                # 4. append to csv log
                log_row = [filename_xyz, algorithm, iteration, sub_iteration, is_initial, is_elite,
                           smiles, sa_score, clogp_score, qed_score, combined_score]
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
        print(f"Improvement over Baseline: {best_score - baseline_score:.4f} ({(best_score - baseline_score) / baseline_score * 100:.2f}%)")
        print(f"Duration: {duration:.2f} seconds")

        nfe = metadata.get('nfe', 0)
        print(f"Total Evaluations (NFE): {nfe}")

        best_sa_score = None
        best_clogp_score = None
        best_qed_score = None
        if best_sample is not None:
            try:
                best_sa_score = sa_verifier(best_sample)
                best_clogp_score = clogp_verifier(best_sample)
                best_qed_score = qed_verifier(best_sample)
                print(f"Best Individual Scores - SA: {best_sa_score:.4f}, cLogP: {best_clogp_score:.4f}, QED: {best_qed_score:.4f}")
            except Exception as e:
                logging.warning(f"Could not calculate individual scores for best sample: {e}")

        # update results dictionary with NFE and individual scores
        results.update({
            'nfe': nfe,
            'best_sa_score': best_sa_score,
            'best_clogp_score': best_clogp_score,
            'best_qed_score': best_qed_score
        })

        save_results(results, output_path)
        
        if args.plot:
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
    run_experiment(args) 