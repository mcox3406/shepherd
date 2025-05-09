#!/bin/bash

# Search Over Paths Parameter Sweep Runner
# Submits a single Slurm job that runs all experiments

cd /home/kento/projects/shepherd/shepherd-inference/examples/inference_scaling

# --- Configuration --- 
OUTPUT_BASE_DIR="inference_scaling_experiments/search_over_paths"
LOG_DIR="sweep_logs"
# ---------------------------------

# create directories for logs and results if they don't exist
mkdir -p "${OUTPUT_BASE_DIR}"
mkdir -p "${LOG_DIR}"

# Generate a unique job name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="sop_sweep_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

echo "Submitting Search Over Paths sweep job..."

# Submit to Slurm
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=10:00:00
#SBATCH --partition=mit_normal_gpu

# Load any necessary modules
module load cuda/12.4.0

# Set PYTHONUNBUFFERED to ensure Python output is not buffered
export PYTHONUNBUFFERED=1

# Run the sweep script. Slurm will handle output redirection.
python run_sop_sweep.py
EOF

echo "Job submitted. Monitor progress via:"
echo "  - Log file: ${LOG_FILE}"
echo "  - Results: ${OUTPUT_BASE_DIR}"
echo "  - Job status: squeue -u \$USER" 