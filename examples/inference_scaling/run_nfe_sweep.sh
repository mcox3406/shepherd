#!/bin/bash

# NFE Sweep Experiment Runner
# Runs Random, Zero-Order, and Guided search with varying NFE targets,
# distributing jobs across available GPUs

cd /home/mcox340/shepherd/examples/inference_scaling

# --- Configuration --- 
CHECKPOINT_PATH="../../data/shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt"
OUTPUT_BASE_DIR="inference_scaling_experiments"
SA_WEIGHT=1.0
CLOGP_WEIGHT=0.0
QED_WEIGHT=0.0
NUM_GPUS=4
# ---------------------------------

# create directories for logs and results if they don't exist
LOG_DIR="sweep_logs"
mkdir -p "${OUTPUT_BASE_DIR}"
mkdir -p "${LOG_DIR}"

# base shared arguments (device will be added per job)
BASE_SHARED_ARGS=(
    --checkpoint "${CHECKPOINT_PATH}"
    --output_dir "${OUTPUT_BASE_DIR}"
    --sa_weight "${SA_WEIGHT}"
    --clogp_weight "${CLOGP_WEIGHT}"
    --qed_weight "${QED_WEIGHT}"
    --n_atoms 40
    --n_pharm 10
)

# job counter for GPU assignment
job_count=0

echo "Starting NFE Sweep experiments... Logs will be in ${LOG_DIR}"
echo "Distributing across ${NUM_GPUS} GPUs."

# --- Random Search --- 
# NFE = number of trials
ALG="random"
NFE_TARGETS=(100)
for NFE in "${NFE_TARGETS[@]}"; do
    gpu_idx=$((job_count % NUM_GPUS))
    EXP_NAME="${ALG}_nfe${NFE}_gpu${gpu_idx}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    echo "Launching: ${EXP_NAME} on GPU ${gpu_idx}"
    
    JOB_ARGS=("${BASE_SHARED_ARGS[@]}" --device "cuda:${gpu_idx}")

    nohup python run_inference_scaling_experiment.py \
        "${JOB_ARGS[@]}" \
        --algorithm "${ALG}" \
        --num_trials "${NFE}" \
        --exp_name "${EXP_NAME}" \
        > "${LOG_FILE}" 2>&1 &
    job_count=$((job_count + 1))
    sleep 1
done

# --- Zero-Order Search --- 
# NFE ≈ 1 (initial sample) + (NUM_STEPS × N_NEIGHBORS)
ALG="zero_order"
N_NEIGHBORS=10
STEP_SIZE=0.1
# pairs of [NFE_TARGET, NUM_STEPS]
NFE_STEPS_PAIRS=(
    "50 5"
    "100 10"
)
for pair in "${NFE_STEPS_PAIRS[@]}"; do
    read -r NFE NUM_STEPS <<< "${pair}"
    gpu_idx=$((job_count % NUM_GPUS))
    EXP_NAME="${ALG}_nfe${NFE}_s${NUM_STEPS}_n${N_NEIGHBORS}_gpu${gpu_idx}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    echo "Launching: ${EXP_NAME} on GPU ${gpu_idx}"

    JOB_ARGS=("${BASE_SHARED_ARGS[@]}" --device "cuda:${gpu_idx}")

    nohup python run_inference_scaling_experiment.py \
        "${JOB_ARGS[@]}" \
        --algorithm "${ALG}" \
        --num_steps "${NUM_STEPS}" \
        --num_neighbors "${N_NEIGHBORS}" \
        --step_size "${STEP_SIZE}" \
        --exp_name "${EXP_NAME}" \
        > "${LOG_FILE}" 2>&1 &
    job_count=$((job_count + 1))
    sleep 1
done

# --- Guided Search --- 
# NFE ≈ POP_SIZE (initial population) + (NUM_GENS × (POP_SIZE - elite count))
ALG="guided"
POP_SIZE=10
ELITE_FRAC=0.2
MUT_RATE=0.2
# pairs of [NFE_TARGET, NUM_GENS]
NFE_GENS_PAIRS=(
    "80 10"
    "160 20"
)
for pair in "${NFE_GENS_PAIRS[@]}"; do
    read -r NFE NUM_GENS <<< "${pair}"
    gpu_idx=$((job_count % NUM_GPUS))
    EXP_NAME="${ALG}_nfe${NFE}_p${POP_SIZE}_g${NUM_GENS}_e${ELITE_FRAC}_gpu${gpu_idx}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    echo "Launching: ${EXP_NAME} on GPU ${gpu_idx}"

    JOB_ARGS=("${BASE_SHARED_ARGS[@]}" --device "cuda:${gpu_idx}")

    nohup python run_inference_scaling_experiment.py \
        "${JOB_ARGS[@]}" \
        --algorithm "${ALG}" \
        --pop_size "${POP_SIZE}" \
        --num_generations "${NUM_GENS}" \
        --mutation_rate "${MUT_RATE}" \
        --elite_fraction "${ELITE_FRAC}" \
        --exp_name "${EXP_NAME}" \
        > "${LOG_FILE}" 2>&1 &
    job_count=$((job_count + 1))
    sleep 1
done

echo "All ${job_count} NFE sweep jobs launched in the background."
echo "Monitor progress via log files in ${LOG_DIR} and check results in ${OUTPUT_BASE_DIR}." 