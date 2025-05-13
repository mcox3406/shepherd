#!/usr/bin/env python3
"""
Run SearchOverPaths parameter sweep experiments.
This script handles running multiple experiments with different N and M values.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging
from datetime import datetime
import select
import fcntl

# Configure logging to console (Slurm will capture this)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(log_file), # Removed: Slurm will handle file logging
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
# Slurm will capture stdout/stderr into the log file defined in the submission script.
logger.info(f"Starting SearchOverPaths sweep experiment. Output will be captured by Slurm.")

def run_experiment(args):
    """Run a single experiment with the given parameters."""
    cmd = [
        "python", "-u", "run_inference_scaling_experiment.py",
        "--checkpoint", args["checkpoint"],
        "--output_dir", args["output_dir"],
        "--algorithm", "search_over_paths",
        "--num_paths_N", str(args["N"]),
        "--path_width_M", str(args["M"]),
        "--initial_t_idx", str(args["initial_t_idx"]),
        "--delta_f", str(args["delta_f"]),
        "--delta_b", str(args["delta_b"]),
        "--n_atoms", str(args["n_atoms"]),
        "--n_pharm", str(args["n_pharm"]),
        "--sa_weight", str(args["sa_weight"]),
        "--clogp_weight", str(args["clogp_weight"]),
        "--device", "cuda:0",
        "--exp_name", args["exp_name"]
    ]
    
    logger.info(f"Running experiment: {args['exp_name']}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the process and capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Make file descriptors non-blocking
        for pipe in [process.stdout, process.stderr]:
            fcntl.fcntl(pipe.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
        
        # Read output in real-time
        while True:
            # Check if process has finished
            if process.poll() is not None:
                break
                
            # Check for output on both stdout and stderr
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [], 0.1)
            
            if ret[0]:
                # Read from stdout
                try:
                    output = process.stdout.readline()
                    if output:
                        logger.info(f"[{args['exp_name']}] {output.strip()}")
                except IOError:
                    pass
                    
                # Read from stderr
                try:
                    error = process.stderr.readline()
                    if error:
                        logger.error(f"[{args['exp_name']}] {error.strip()}")
                except IOError:
                    pass
        
        # Get any remaining output
        stdout, stderr = process.communicate()
        
        # Log any remaining stdout
        if stdout:
            for line in stdout.splitlines():
                logger.info(f"[{args['exp_name']}] {line}")
        
        # Log any stderr
        if stderr:
            for line in stderr.splitlines():
                logger.error(f"[{args['exp_name']}] {line}")
        
        # Check return code
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
        logger.info(f"Experiment {args['exp_name']} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment {args['exp_name']} failed with error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in experiment {args['exp_name']}: {e}")
        return False

def main():
    # Base configuration
    base_config = {
        # Model and output configuration
        "checkpoint": "../../data/shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt",
        "output_dir": "inference_scaling_experiments/search_over_paths/multi_objective",
        "algorithm": "search_over_paths",  # Specify the search algorithm
        "verbose": True,
        
        # SearchOverPaths specific parameters
        "initial_t_idx": 355,  # Initial time to denoise to
        "delta_f": 312,        # Number of steps to forward noise
        "delta_b": 324,        # Number of steps to reverse noise (must be > delta_f)
        
        # Model configuration
        "n_atoms": 40,         # Number of atoms to generate
        "n_pharm": 10,          # Number of pharmacophores
        "sa_weight": 1.0,      # Weight for synthetic accessibility score
        "clogp_weight": 1.0,   # Weight for cLogP score
        
        # Sampler configuration
        "sampler_type": "ddpm",  # Sampling algorithm (ddpm or ddim)
        "num_sampling_steps": None,  # Number of steps for the sampler
        "ddim_eta": 0.0,       # Eta parameter for DDIM sampling
        
        # Inference configuration
        "batch_size": 1,       # Batch size for inference
        "device": "cuda:0",    # Device to run inference on
        "verbose": True,       # Print detailed progress information
        "plot": True,         # Generate plots after the experiment
        "save_checkpoint": False,  # Save checkpoints during search
        "checkpoint_interval": 10  # Interval for saving checkpoints
    }
    
    # Parameter sweep values
    N_values = [2, 8]  # Number of paths
    M_values = [15, 5]  # Number of samples per path
    
    # Create output directory
    output_dir = Path(base_config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a results summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"sop_sweep_summary_{timestamp}.json"
    results = []
    
    logger.info(f"Starting parameter sweep with N={N_values}, M={M_values}")
    logger.info(f"Results will be saved to: {summary_file}")
    
    # Run experiments
    for N in N_values:
        for M in M_values:
            exp_name = f"search_over_paths_N{N}_M{M}"
            
            # Calculate expected NFE
            nfe = N * M * 4  # Assuming ~4 iterations per path
            
            # Create experiment config
            exp_config = base_config.copy()
            exp_config.update({
                "N": N,
                "M": M,
                "exp_name": exp_name,
                "expected_nfe": nfe
            })
            
            # Run experiment
            success = run_experiment(exp_config)
            
            # Record results
            result = {
                "experiment": exp_name,
                "N": N,
                "M": M,
                "expected_nfe": nfe,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "config": exp_config  # Include full configuration in results
            }
            results.append(result)
            
            # Save intermediate results
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Completed experiment {exp_name} (N={N}, M={M}, NFE={nfe})")
    
    logger.info(f"All experiments completed. Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 