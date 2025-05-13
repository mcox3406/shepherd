"""
Run Classifier-Free Guidance (CFG) sweep experiments.
This script handles running multiple experiments with different CFG weights
and NFE targets for random and zero-order search algorithms.
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
import math
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(args):
    """Run a single experiment with the given parameters."""
    cmd = [
        "python", "-u", "run_inference_scaling_experiment.py",
        "--checkpoint", args["checkpoint"],
        "--output_dir", args["output_dir"],
        "--algorithm", args["algorithm"],
        "--n_atoms", str(args["n_atoms"]),
        "--n_pharm", str(args["n_pharm"]),
        "--sa_weight", str(args["sa_weight"]),
        "--clogp_weight", str(args["clogp_weight"]),
        "--device", args["device"],
        "--exp_name", args["exp_name"],
        "--batch_size", str(args["batch_size"]),
        # CFG specific
        "--sa_score_target_value", str(args["sa_score_target_value"])
    ]

    if args.get("do_property_cfg", False):
        cmd.append("--do_property_cfg")
        cmd.extend(["--cfg_weight", str(args["cfg_weight"])])

    # Algorithm specific NFE params
    if args["algorithm"] == "random":
        cmd.extend(["--num_trials", str(args["num_trials"])])
    elif args["algorithm"] == "zero_order":
        cmd.extend([
            "--num_steps", str(args["num_steps"]),
            "--num_neighbors", str(args["num_neighbors"]),
            "--step_size", str(args["step_size"])
        ])
    
    logger.info(f"Running experiment: {args['exp_name']}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for pipe in [process.stdout, process.stderr]:
            fcntl.fcntl(pipe.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
        
        while True:
            if process.poll() is not None:
                break
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [], 0.1)
            if ret[0]:
                try:
                    output = process.stdout.readline()
                    if output:
                        logger.info(f"[{args['exp_name']}] {output.strip()}")
                except IOError:
                    pass
                try:
                    error = process.stderr.readline()
                    if error:
                        logger.error(f"[{args['exp_name']}] {error.strip()}")
                except IOError:
                    pass
        
        stdout, stderr = process.communicate()
        if stdout:
            for line in stdout.splitlines():
                logger.info(f"[{args['exp_name']}] {line}")
        if stderr:
            for line in stderr.splitlines():
                logger.error(f"[{args['exp_name']}] {line}")
        
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
    parser = argparse.ArgumentParser(description="Run CFG parameter sweep experiments.")
    parser.add_argument(
        "--algorithm_type",
        type=str,
        default="all",
        choices=["all", "random", "zero_order"],
        help="Type of algorithm to run experiments for (all, random, zero_order)."
    )
    script_args = parser.parse_args()
    logger.info(f"Running CFG sweep for algorithm type: {script_args.algorithm_type}")

    base_config = {
        "checkpoint": "../../data/shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_cfg_finetune.ckpt",
        "output_dir": "inference_scaling_experiments/cfg_sweep3",
        "n_atoms": 40,
        "n_pharm": 10,
        "sa_weight": 1.0,
        "clogp_weight": 0.0,
        "device": "cuda:0", # Default device, can be changed
        "sa_score_target_value": 8.0, # Target for SA score in CFG (reasonably attainable)
        "do_property_cfg": True, # Enable CFG for all experiments in this sweep
        "batch_size": 15,
    }

    # Parameter sweep values
    CFG_WEIGHTS_TO_SWEEP = [0.0, 3.0]
    
    ALL_ALGORITHMS_CONFIG = { # Renamed to avoid conflict
        "random": {
            "nfe_targets": [500, 100, 50],
        },
        "zero_order": {
            "nfe_targets": [500, 100, 50], 
            "num_neighbors": 10,
            "step_size": 0.1,
        }
    }

    # Filter algorithms based on the script argument
    if script_args.algorithm_type == "all":
        algorithms_to_run = ALL_ALGORITHMS_CONFIG
    elif script_args.algorithm_type == "random":
        algorithms_to_run = {"random": ALL_ALGORITHMS_CONFIG["random"]}
    elif script_args.algorithm_type == "zero_order":
        algorithms_to_run = {"zero_order": ALL_ALGORITHMS_CONFIG["zero_order"]}
    else: # Should not happen due to choices in argparse
        logger.error(f"Invalid algorithm type: {script_args.algorithm_type}")
        sys.exit(1)
        
    output_dir_base = Path(base_config["output_dir"])
    output_dir_base.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include algorithm type in summary file name for clarity
    summary_file_name = f"cfg_sweep_{script_args.algorithm_type}_summary_{timestamp}.json"
    summary_file = output_dir_base / summary_file_name
    results_summary = []
    
    logger.info(f"Starting CFG parameter sweep for {script_args.algorithm_type} algorithms.")
    logger.info(f"Results will be saved to: {summary_file}")

    for cfg_w in CFG_WEIGHTS_TO_SWEEP:
        for alg_name, alg_config in algorithms_to_run.items(): # Use filtered dict
            for nfe_target in alg_config["nfe_targets"]:
                
                exp_config = base_config.copy()
                exp_config["algorithm"] = alg_name
                exp_config["cfg_weight"] = cfg_w
                
                exp_name_parts = [
                    "cfg",
                    alg_name,
                    f"w{cfg_w}",
                    f"nfe{nfe_target}",
                    f"sa_target{exp_config['sa_score_target_value']}"
                ]
                exp_name = "_".join(exp_name_parts)
                exp_config["exp_name"] = exp_name

                if alg_name == "random":
                    exp_config["num_trials"] = nfe_target
                elif alg_name == "zero_order":
                    # NFE for ZeroOrderSearch ≈ 1 (initial) + num_steps * num_neighbors
                    # So, num_steps ≈ (nfe_target - 1) / num_neighbors
                    num_neighbors = alg_config["num_neighbors"]
                    num_steps = math.ceil((nfe_target -1) / num_neighbors)
                    if num_steps <=0: num_steps = 1 # Ensure at least 1 step
                    
                    exp_config["num_steps"] = num_steps
                    exp_config["num_neighbors"] = num_neighbors
                    exp_config["step_size"] = alg_config["step_size"]
                    # Recalculate actual NFE for more accurate naming/logging if desired
                    actual_nfe = 1 + num_steps * num_neighbors
                    logger.info(f"Zero-Order for NFE target {nfe_target}: using {num_steps} steps, {num_neighbors} neighbors. Approx NFE: {actual_nfe}")


                # Run experiment
                success = run_experiment(exp_config)
                
                # Record results
                result_entry = {
                    "experiment_name": exp_name,
                    "algorithm": alg_name,
                    "cfg_weight": cfg_w,
                    "sa_score_target_value": exp_config["sa_score_target_value"],
                    "nfe_target": nfe_target,
                    "success": success,
                    "timestamp": datetime.now().isoformat(),
                    "full_config": exp_config
                }
                if alg_name == "zero_order":
                    result_entry["num_steps"] = exp_config["num_steps"]
                    result_entry["num_neighbors"] = exp_config["num_neighbors"]
                    result_entry["actual_nfe_approx"] = 1 + exp_config["num_steps"] * exp_config["num_neighbors"]


                results_summary.append(result_entry)
                
                with open(summary_file, 'w') as f:
                    json.dump(results_summary, f, indent=2)
                
                logger.info(f"Completed experiment {exp_name}")
    
    logger.info(f"All CFG sweep experiments for {script_args.algorithm_type} completed. Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 