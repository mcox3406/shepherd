"""
Basic inference test script to see if we can load the model and generate molecules.
This script is intended for debugging purposes.
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import pickle

from shepherd.lightning_module import LightningModule
from shepherd.inference import inference_sample

from rdkit import Chem
from rdkit.Chem import Draw

def parse_args():
    parser = argparse.ArgumentParser(description='Basic ShEPhERD inference test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cpu, cuda, mps). Default: auto-select best device')
    parser.add_argument('--output_dir', type=str, default='basic_inference_results',
                        help='Directory to save results')
    parser.add_argument('--n_atoms', type=int, default=8,
                        help='Number of atoms to generate')
    parser.add_argument('--n_pharm', type=int, default=2,
                        help='Number of pharmacophores')
    parser.add_argument('--fast', action='store_true',
                        help='Run in fast mode with fewer timesteps')
    return parser.parse_args()

def get_device(device_arg):
    """Select the best available device"""
    if device_arg is not None:
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS is available, but we are not using it because it is not supported by torch_cluster")
        return torch.device('cpu')
    else:
        return torch.device('cpu')

def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # get the best available device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.checkpoint}")
    
    try:
        # load the model
        model_pl = LightningModule.load_from_checkpoint(args.checkpoint)
        model_pl.eval()
        model_pl.to(device)
        model_pl.model.device = device
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    params = model_pl.params
    
    # setting inference parameters
    batch_size = 10
    n_atoms = args.n_atoms
    n_pharm = args.n_pharm
    
    if args.fast:
        # print("Running in fast mode with fewer steps")
        # I couldn't get this to work for the moment, but it doesn't matter
        pass
    
    T = params['noise_schedules']['x1']['ts'].max()
    inject_noise_at_ts = list(np.arange(130, 80, -1))
    inject_noise_scales = [1.0] * len(inject_noise_at_ts)
    harmonize = True
    harmonize_ts = [80]
    harmonize_jumps = [20]
    
    print(f"Running inference with {n_atoms} atoms and {n_pharm} pharmacophores on {device}")
    print(f"Number of timesteps in noise schedule: {len(params['noise_schedules']['x1']['ts'])}")
    
    # run inference
    try:
        samples = inference_sample(
            model_pl,
            batch_size=batch_size,
            
            N_x1=n_atoms,
            N_x4=n_pharm,
            
            unconditional=True,
            
            prior_noise_scale=1.0,
            denoising_noise_scale=1.0,
            
            # noise injection parameters
            inject_noise_at_ts=inject_noise_at_ts,
            inject_noise_scales=inject_noise_scales,
            harmonize=harmonize,
            harmonize_ts=harmonize_ts,
            harmonize_jumps=harmonize_jumps,
            
            # inpainting parameters (not used for unconditional generation)
            inpaint_x2_pos=False,
            inpaint_x3_pos=False,
            inpaint_x3_x=False,
            inpaint_x4_pos=False,
            inpaint_x4_direction=False,
            inpaint_x4_type=False,
            
            stop_inpainting_at_time_x2=0.0,
            add_noise_to_inpainted_x2_pos=0.0,
            
            stop_inpainting_at_time_x3=0.0,
            add_noise_to_inpainted_x3_pos=0.0,
            add_noise_to_inpainted_x3_x=0.0,
            
            stop_inpainting_at_time_x4=0.0,
            add_noise_to_inpainted_x4_pos=0.0,
            add_noise_to_inpainted_x4_direction=0.0,
            add_noise_to_inpainted_x4_type=0.0,
            
            # inpainting targets (not used for unconditional generation)
            # IMPORTANT: we need to match the dimensions of these arrays to our n_pharm value
            center_of_mass=np.zeros(3),
            surface=np.zeros((75, 3)),
            electrostatics=np.zeros(75),
            pharm_types=np.zeros(n_pharm, dtype=int),
            pharm_pos=np.zeros((n_pharm, 3)), 
            pharm_direction=np.zeros((n_pharm, 3)),
        )
        
        print("Inference completed successfully!")
        
        # save samples to pickle file
        with open(output_dir / "samples.pickle", "wb") as f:
            pickle.dump(samples, f)
        print(f"Saved samples to {output_dir}/samples.pickle")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
