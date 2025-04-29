# Inference Scaling Experiments for ShEPhERD

This directory contains example scripts for running and analyzing inference-time scaling experiments with ShEPhERD.

## Workflow

1.  **Run an Experiment:** Use `run_inference_scaling_experiment.py` to run a specific search algorithm.
2.  **Analyze Multiple Experiments:** Use `analyze_inference_scaling_experiments.py` to compare summary results across several experiment runs.
3.  **Visualize Detailed Results:** Use `visualize_scaling_results.py` to generate plots from the detailed log of a single experiment.

## Usage

**1. Running a Basic Experiment**

To run a single experiment, use the `run_inference_scaling_experiment.py` script. You need to specify a model checkpoint and the desired search algorithm.

```bash
# Example: Run Random Search for 50 trials
python examples/inference_scaling/run_inference_scaling_experiment.py \
    --checkpoint /path/to/your/shepherd/checkpoint.ckpt \
    --algorithm random \
    --num_trials 50 \
    --sa_weight 1.0 \
    --clogp_weight 1.0 \
    --output_dir examples/inference_scaling/inference_scaling_experiments \
    --exp_name random_search_example \
    --verbose
```

*   Replace `/path/to/your/shepherd/checkpoint.ckpt` with the actual path.
*   Supported algorithms: `random`, `zero_order`, `guided`.
*   Adjust parameters (e.g., `--num_trials`, `--num_steps`, `--pop_size`) based on the chosen algorithm.
*   Results, including the detailed log (`all_molecules_log.csv`) and XYZ files (`all_molecules/`), will be saved in the specified output directory under the experiment name (e.g., `examples/inference_scaling/inference_scaling_experiments/random_search_example`).

**2. Analyzing Multiple Experiments**

To compare the summary results (best scores, duration, etc.) from multiple experiments stored in the `inference_scaling_experiments` directory:

```bash
python examples/inference_scaling/analyze_inference_scaling_experiments.py \
    --base_dir examples/inference_scaling/inference_scaling_experiments \
    --draw_molecules
```

*   This script will look for experiment subdirectories within the `--base_dir`.
*   It prints a summary table and saves comparison plots and molecule images (if `--draw_molecules` is used) to an `analysis` subdirectory within the `--base_dir`.

**3. Visualizing Detailed Results**

To generate detailed plots (score distributions, progress, trade-offs, diversity) from the `all_molecules_log.csv` of a *single* experiment:

```bash
python examples/inference_scaling/visualize_scaling_results.py \
    examples/inference_scaling/inference_scaling_experiments/random_search_example 
```

*   Replace `random_search_example` with the specific experiment directory name.
*   Plots will be saved into an `analysis` subdirectory within that specific experiment's directory. 