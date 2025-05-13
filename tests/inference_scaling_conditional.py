"""
Integration tests for inference scaling with ShEPhERD.

This script demonstrates how to use the inference scaling components with
a real ShEPhERD model. It can be run as a standalone script to validate
that all components work together correctly.
"""

import os
import sys
import torch
import numpy as np
import argparse
import pickle
import rdkit
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from shepherd.lightning_module import LightningModule
from src.shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atom_coords, 
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics,
    get_electrostatics_given_point_charges,
)
from src.shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from src.shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.inference_scaling import (
    ShepherdModelRunner,
    SAScoreVerifier,
    CLogPVerifier,
    MultiObjectiveVerifier,
    RandomSearch,
    ZeroOrderSearch
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test inference scaling with ShEPhERD')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cpu, cuda, mps). Default: auto-select best device')
    parser.add_argument('--output_dir', type=str, default='inference_scaling_test_results',
                        help='Directory to save results')
    parser.add_argument('--conf_file', type=str, default=None,
                        help='Path to conformers file to extract interaction profiles for conditional generation')
    parser.add_argument('--conf_index', type=int, default=0,
                        help='Index of conformer to use for conditional generation')
    parser.add_argument('--n_atoms', type=int, default=25,
                        help='Number of atoms to generate')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of trials for random search')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--fast', action='store_true',
                        help='Run with limited trials and steps for quick testing')
    # checkpoint control arguments
    parser.add_argument('--start_step', type=int, default=1, choices=[1, 2, 3],
                       help='Step to start from (1=baseline, 2=random search, 3=zero-order search)')
    parser.add_argument('--use_cached', action='store_true',
                       help='Use cached results from previous runs if available')
    parser.add_argument('--skip_plots', action='store_true',
                       help='Skip generating plots')
    return parser.parse_args()


def get_device(device_arg):
    """Select the best available device"""
    if device_arg is not None:
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS is available, but we are not using it because it may not be fully supported")
        return torch.device('cpu')
    else:
        return torch.device('cpu')


def save_results(results, output_dir, filename):
    """Save results to a pickle file for later use."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {output_dir / filename}")


def load_results(output_dir, filename):
    """Load results from a pickle file if it exists."""
    filepath = Path(output_dir) / filename
    if filepath.exists():
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def get_conditional_info(args, params):
    with open(args.conf_file, 'rb') as f:
        molblocks_and_charges = pickle.load(f)
    
    index = args.conf_index
    mol = rdkit.Chem.MolFromMolBlock(molblocks_and_charges[index][0], removeHs = False) # target natural product
    charges = np.array(molblocks_and_charges[index][1]) # xTB partial charges in implicit water
    #display(mol)

    # extracting target interaction profiles (ESP and pharmacophores)
    mol_coordinates = np.array(mol.GetConformer().GetPositions())
    mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)
    mol = update_mol_coordinates(mol, mol_coordinates)

    # conditional targets
    centers = mol.GetConformer().GetPositions()
    radii = get_atomic_vdw_radii(mol)
    surface = get_molecular_surface(
        centers, 
        radii, 
        params['dataset']['x3']['num_points'], 
        probe_radius = params['dataset']['probe_radius'],
        num_samples_per_atom = 20,
    )

    pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
        mol,
        multi_vector = params['dataset']['x4']['multivectors'],
        check_access = params['dataset']['x4']['check_accessibility'],
    )

    electrostatics = get_electrostatics_given_point_charges(
        charges, centers, surface,
    )
    return surface, pharm_types, pharm_pos, pharm_direction, electrostatics

def run_inference_scaling_test(args):
    zo_num_steps = 100
    """Run test of inference scaling components with a real model."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    results = {}
    
    # Step 1: Generate a baseline sample
    if args.start_step <= 1:
        baseline_results = None
        if args.use_cached:
            baseline_results = load_results(output_dir, "baseline_results.pkl")
        
        if baseline_results is None:
            print(f"Loading model from {args.checkpoint}")
            try:
                model_pl = LightningModule.load_from_checkpoint(args.checkpoint)
                model_pl.eval()
                model_pl.to(device)
                model_pl.model.device = device
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
            
            surface, pharm_types, pharm_pos, pharm_direction, electrostatics = get_conditional_info(args, model_pl.params)
            n_pharm = len(pharm_types)

            # set up inference parameters - using the same defaults as basic_inference_test.py
            T = model_pl.params['noise_schedules']['x1']['ts'].max()
            inject_noise_at_ts = []
            inject_noise_scales = []
            harmonize = False
            harmonize_ts = []
            harmonize_jumps = []
            
            # validate harmonize parameters to avoid IndexError
            if harmonize and not harmonize_ts:
                print("Warning: harmonize is True but harmonize_ts is empty. Setting harmonize_ts to [80]")
                harmonize_ts = [80]
            if harmonize and not harmonize_jumps:
                print("Warning: harmonize is True but harmonize_jumps is empty. Setting harmonize_jumps to [20]")
                harmonize_jumps = [20]
            
            print(f"Creating model runner for {args.n_atoms} atoms and {n_pharm} pharmacophores")
            model_runner = ShepherdModelRunner(
                model_pl=model_pl,
                batch_size=args.batch_size,
                N_x1=args.n_atoms,
                N_x4=n_pharm,
                unconditional=False,
                device=str(device),
                save_all_noise=True,
                
                # these are the same parameters used in basic_inference_test.py
                prior_noise_scale=1.0,
                denoising_noise_scale=1.0,
                inject_noise_at_ts=inject_noise_at_ts,
                inject_noise_scales=inject_noise_scales,
                harmonize=harmonize,
                harmonize_ts=harmonize_ts,
                harmonize_jumps=harmonize_jumps,
                inpaint_x2_pos = False, # note that x2 is implicitly modeled via x3
    
                inpaint_x3_pos = True,
                inpaint_x3_x = True,
                
                inpaint_x4_pos = True,
                inpaint_x4_direction = True,
                inpaint_x4_type = True,
                
                stop_inpainting_at_time_x2 = 0.0,
                add_noise_to_inpainted_x2_pos = 0.0,
                
                stop_inpainting_at_time_x3 = 0.0,
                add_noise_to_inpainted_x3_pos = 0.0,
                add_noise_to_inpainted_x3_x = 0.0,
                
                stop_inpainting_at_time_x4 = 0.0,
                add_noise_to_inpainted_x4_pos = 0.0,
                add_noise_to_inpainted_x4_direction = 0.0,
                add_noise_to_inpainted_x4_type = 0.0,
                
                # these are the inpainting targets
                center_of_mass = np.zeros(3), # center of mass of x1; already centered to zero above
                surface = surface,
                electrostatics = electrostatics,
                pharm_types = pharm_types,
                pharm_pos = pharm_pos,
                pharm_direction = pharm_direction,
            )
            
            print("Creating verifiers")
            sa_verifier = SAScoreVerifier(weight=1.0)
            clogp_verifier = CLogPVerifier(weight=0.0)  # disable cLogP verifier
            multi_verifier = MultiObjectiveVerifier([sa_verifier, clogp_verifier])
            
            print("\n=== Step 1: Generate baseline sample ===")
            baseline_sample = model_runner()
            
            baseline_noise = model_runner.get_last_noise()
            
            # evaluate baseline sample
            try:
                baseline_score_sa = sa_verifier(baseline_sample)
                baseline_score_clogp = clogp_verifier(baseline_sample)
                baseline_score_multi = multi_verifier(baseline_sample)
                
                print(f"Baseline SA Score: {baseline_score_sa:.4f}")
                # print(f"Baseline cLogP Score: {baseline_score_clogp:.4f}")
                # print(f"Baseline Multi-Objective Score: {baseline_score_multi:.4f}")
                
                # save baseline results
                baseline_results = {
                    'sample': baseline_sample,
                    'noise': baseline_noise,
                    'sa_score': baseline_score_sa,
                    'clogp_score': baseline_score_clogp,
                    'multi_score': baseline_score_multi
                }
                save_results(baseline_results, output_dir, f"baseline_results_{zo_num_steps}_trials.pkl")
            except Exception as e:
                print(f"Error evaluating baseline sample: {e}")
                return
        else:
            print("\n=== Step 1: Using cached baseline sample ===")
            print(f"Baseline SA Score: {baseline_results['sa_score']:.4f}")
            # print(f"Baseline cLogP Score: {baseline_results['clogp_score']:.4f}")
            # print(f"Baseline Multi-Objective Score: {baseline_results['multi_score']:.4f}")
            
            if args.start_step <= 2:
                print(f"Loading model from {args.checkpoint}")
                try:
                    model_pl = LightningModule.load_from_checkpoint(args.checkpoint)
                    model_pl.eval()
                    model_pl.to(device)
                    model_pl.model.device = device
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    return
                
                surface, pharm_types, pharm_pos, pharm_direction, electrostatics = get_conditional_info(args, model_pl.params)
                n_pharm = len(pharm_types)

                # set up inference parameters
                T = model_pl.params['noise_schedules']['x1']['ts'].max()
                inject_noise_at_ts = []
                inject_noise_scales = []
                harmonize = False
                harmonize_ts = []
                harmonize_jumps = []
                
                # validate harmonize parameters to avoid IndexError
                if harmonize and not harmonize_ts:
                    print("Warning: harmonize is True but harmonize_ts is empty. Setting harmonize_ts to [80]")
                    harmonize_ts = [80]
                if harmonize and not harmonize_jumps:
                    print("Warning: harmonize is True but harmonize_jumps is empty. Setting harmonize_jumps to [20]")
                    harmonize_jumps = [20]
                
                print(f"Creating model runner for next steps")
                model_runner = ShepherdModelRunner(
                    model_pl=model_pl,
                    batch_size=1,
                    N_x1=args.n_atoms,
                    N_x4=n_pharm,
                    unconditional=False,
                    device=str(device),
                    save_all_noise=True,
                    prior_noise_scale=1.0,
                    denoising_noise_scale=1.0,
                    inject_noise_at_ts=inject_noise_at_ts,
                    inject_noise_scales=inject_noise_scales,
                    harmonize=harmonize,
                    harmonize_ts=harmonize_ts,
                    harmonize_jumps=harmonize_jumps,
                    inpaint_x2_pos = False, # note that x2 is implicitly modeled via x3
    
                    inpaint_x3_pos = True,
                    inpaint_x3_x = True,
                    
                    inpaint_x4_pos = True,
                    inpaint_x4_direction = True,
                    inpaint_x4_type = True,
                    
                    stop_inpainting_at_time_x2 = 0.0,
                    add_noise_to_inpainted_x2_pos = 0.0,
                    
                    stop_inpainting_at_time_x3 = 0.0,
                    add_noise_to_inpainted_x3_pos = 0.0,
                    add_noise_to_inpainted_x3_x = 0.0,
                    
                    stop_inpainting_at_time_x4 = 0.0,
                    add_noise_to_inpainted_x4_pos = 0.0,
                    add_noise_to_inpainted_x4_direction = 0.0,
                    add_noise_to_inpainted_x4_type = 0.0,
                    
                    # these are the inpainting targets
                    center_of_mass = np.zeros(3), # center of mass of x1; already centered to zero above
                    surface = surface,
                    electrostatics = electrostatics,
                    pharm_types = pharm_types,
                    pharm_pos = pharm_pos,
                    pharm_direction = pharm_direction,
                )
                
                print("Creating verifiers")
                sa_verifier = SAScoreVerifier(weight=1.0)
                clogp_verifier = CLogPVerifier(weight=1.0)
                multi_verifier = MultiObjectiveVerifier([sa_verifier, clogp_verifier])
        
        results['baseline'] = baseline_results
    else:
        # load baseline results for comparison
        baseline_results = load_results(output_dir, "baseline_results.pkl")
        if baseline_results is None:
            print("Error: Cannot start from step > 1 without baseline results")
            print("Please run with --start_step 1 first")
            return
        
        print("\n=== Skipping Step 1 (baseline generation) ===")
        print(f"Using cached baseline results:")
        print(f"Baseline SA Score: {baseline_results['sa_score']:.4f}")
        # print(f"Baseline cLogP Score: {baseline_results['clogp_score']:.4f}")
        # print(f"Baseline Multi-Objective Score: {baseline_results['multi_score']:.4f}")
        
        results['baseline'] = baseline_results
        
        if args.start_step <= 2:
            print(f"Loading model from {args.checkpoint}")
            try:
                model_pl = LightningModule.load_from_checkpoint(args.checkpoint)
                model_pl.eval()
                model_pl.to(device)
                model_pl.model.device = device
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
            
            surface, pharm_types, pharm_pos, pharm_direction, electrostatics = get_conditional_info(args, model_pl.params)
            n_pharm = len(pharm_types)

            # set up inference parameters
            T = model_pl.params['noise_schedules']['x1']['ts'].max()
            inject_noise_at_ts = []
            inject_noise_scales = []
            harmonize = False
            harmonize_ts = []
            harmonize_jumps = []
            
            # validate harmonize parameters to avoid IndexError
            if harmonize and not harmonize_ts:
                print("Warning: harmonize is True but harmonize_ts is empty. Setting harmonize_ts to [80]")
                harmonize_ts = [80]
            if harmonize and not harmonize_jumps:
                print("Warning: harmonize is True but harmonize_jumps is empty. Setting harmonize_jumps to [20]")
                harmonize_jumps = [20]
            
            print(f"Creating model runner for next steps")
            model_runner = ShepherdModelRunner(
                model_pl=model_pl,
                batch_size=1,
                N_x1=args.n_atoms,
                N_x4=n_pharm,
                unconditional=False,
                device=str(device),
                save_all_noise=True,
                prior_noise_scale=1.0,
                denoising_noise_scale=1.0,
                inject_noise_at_ts=inject_noise_at_ts,
                inject_noise_scales=inject_noise_scales,
                harmonize=harmonize,
                harmonize_ts=harmonize_ts,
                harmonize_jumps=harmonize_jumps,
                inpaint_x2_pos = False, # note that x2 is implicitly modeled via x3
    
                inpaint_x3_pos = True,
                inpaint_x3_x = True,
                
                inpaint_x4_pos = True,
                inpaint_x4_direction = True,
                inpaint_x4_type = True,
                
                stop_inpainting_at_time_x2 = 0.0,
                add_noise_to_inpainted_x2_pos = 0.0,
                
                stop_inpainting_at_time_x3 = 0.0,
                add_noise_to_inpainted_x3_pos = 0.0,
                add_noise_to_inpainted_x3_x = 0.0,
                
                stop_inpainting_at_time_x4 = 0.0,
                add_noise_to_inpainted_x4_pos = 0.0,
                add_noise_to_inpainted_x4_direction = 0.0,
                add_noise_to_inpainted_x4_type = 0.0,
                
                # these are the inpainting targets
                center_of_mass = np.zeros(3), # center of mass of x1; already centered to zero above
                surface = surface,
                electrostatics = electrostatics,
                pharm_types = pharm_types,
                pharm_pos = pharm_pos,
                pharm_direction = pharm_direction,
            )
            
            # create verifiers
            print("Creating verifiers")
            sa_verifier = SAScoreVerifier(weight=1.0)
            clogp_verifier = CLogPVerifier(weight=1.0)
            multi_verifier = MultiObjectiveVerifier([sa_verifier, clogp_verifier])
    
    # set number of trials/steps based on --fast flag
    num_trials = 3 if args.fast else args.num_trials
    num_steps = 2 if args.fast else zo_num_steps
    num_neighbors = 2 if args.fast else 3
    
    # Step 2: Run Random Search
    if args.start_step <= 2:
        # try to load cached random search results if requested
        random_search_results = None
        if args.use_cached:
            random_search_results = load_results(output_dir, "random_search_results.pkl")
        
        if random_search_results is None:
            print(f"\n=== Step 2: Run Random Search ({num_trials} trials) ===")
            random_search = RandomSearch(multi_verifier, model_runner)
            random_best_sample, random_best_score, random_scores, random_metadata = random_search.search(
                num_trials=num_trials, 
                verbose=True,
                device=str(device)
            )
            
            print(f"Random Search Best Score: {random_best_score:.4f}")
            print(f"Random Search Score Improvement: {random_best_score - baseline_results['multi_score']:.4f}")
            print(f"Random Search Mean Time per Trial: {random_metadata['mean_time_per_trial']:.2f}s")
            
            # save random search results
            random_search_results = {
                'best_sample': random_best_sample,
                'best_score': random_best_score,
                'scores': random_scores,
                'metadata': random_metadata
            }
            save_results(random_search_results, output_dir, f"random_search_results_{zo_num_steps}_trials.pkl")
        else:
            print("\n=== Step 2: Using cached Random Search results ===")
            print(f"Random Search Best Score: {random_search_results['best_score']:.4f}")
            print(f"Random Search Score Improvement: {random_search_results['best_score'] - baseline_results['multi_score']:.4f}")
            print(f"Random Search Mean Time per Trial: {random_search_results['metadata']['mean_time_per_trial']:.2f}s")
        
        # store random search results
        results['random_search'] = random_search_results
    else:
        # load random search results for comparison
        random_search_results = load_results(output_dir, "random_search_results.pkl")
        if random_search_results is None:
            print("Warning: No cached random search results found, skipping")
        else:
            print("\n=== Skipping Step 2 (Random Search) ===")
            print(f"Using cached Random Search results:")
            print(f"Random Search Best Score: {random_search_results['best_score']:.4f}")
            print(f"Random Search Score Improvement: {random_search_results['best_score'] - baseline_results['multi_score']:.4f}")
            
            results['random_search'] = random_search_results
    
    # Step 3: Run Zero-Order Search
    if args.start_step <= 3:
        # try to load cached zero-order search results if requested
        zo_search_results = None
        if args.use_cached:
            zo_search_results = load_results(output_dir, "zo_search_results.pkl")
        
        if zo_search_results is None:
            print(f"\n=== Step 3: Run Zero-Order Search ({num_steps} steps) ===")
            zo_search = ZeroOrderSearch(multi_verifier, model_runner)
            zo_best_sample, zo_best_score, zo_scores, zo_metadata = zo_search.search(
                num_steps=num_steps,
                num_neighbors=num_neighbors,
                verbose=True,
                device=str(device)
            )
            
            print(f"Zero-Order Search Best Score: {zo_best_score:.4f}")
            print(f"Zero-Order Search Score Improvement: {zo_best_score - baseline_results['multi_score']:.4f}")
            print(f"Zero-Order Search Mean Time per Step: {zo_metadata['mean_time_per_step']:.2f}s")
            
            # save zero-order search results
            zo_search_results = {
                'best_sample': zo_best_sample,
                'best_score': zo_best_score,
                'scores': zo_scores,
                'metadata': zo_metadata
            }
            save_results(zo_search_results, output_dir, f"zo_search_results_{zo_num_steps}_trials.pkl")
        else:
            print("\n=== Step 3: Using cached Zero-Order Search results ===")
            print(f"Zero-Order Search Best Score: {zo_search_results['best_score']:.4f}")
            print(f"Zero-Order Search Score Improvement: {zo_search_results['best_score'] - baseline_results['multi_score']:.4f}")
            print(f"Zero-Order Search Mean Time per Step: {zo_search_results['metadata']['mean_time_per_step']:.2f}s")
        
        # store zero-order search results
        results['zero_order_search'] = zo_search_results
    else:
        print("\n=== Skipping Step 3 (Zero-Order Search) ===")
    
    # generate plots if all required data is available and not skipped
    if not args.skip_plots and 'baseline' in results and 'random_search' in results and 'zero_order_search' in results:
        print("\n=== generating plots ===")
        plt.figure(figsize=(10, 6))
        plt.axhline(y=results['baseline']['multi_score'], color='r', linestyle='-', label='Baseline')
        
        # plot random search scores
        random_scores = results['random_search']['scores']
        random_best_score = results['random_search']['best_score']
        plt.plot(range(1, len(random_scores) + 1), random_scores, 'b-', label='Random Search')
        plt.scatter(len(random_scores), random_best_score, color='b', marker='o')
        
        # plot zero-order search scores
        zo_scores = results['zero_order_search']['scores']
        zo_best_score = results['zero_order_search']['best_score']
        plt.plot(range(1, len(zo_scores) + 1), zo_scores, 'g-', label='Zero-Order Search')
        plt.scatter(len(zo_scores), zo_best_score, color='g', marker='o')
        
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Multi-Objective Score')
        plt.title('Inference Scaling Results')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_dir / "inference_scaling_results.png")
        print(f"Saved plot to {output_dir}/inference_scaling_results.png")

    # for each type of search, write results to csv file (each row is an iteration)
    # for search_type in results.keys():
    #     with open(output_dir / f"{search_type}_results.csv", "w") as f:
    #         f.write("Iteration,Score\n")
    #         for i, score in enumerate(results[search_type]['scores']):
    #             f.write(f"{i+1},{score}\n")

    return results


def main():
    """Main function to run the test."""
    args = parse_args()
    results = run_inference_scaling_test(args)
    


if __name__ == "__main__":
    main() 