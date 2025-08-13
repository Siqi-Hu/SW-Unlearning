#!/bin/bash
#SBATCH --job-name=judgement
#SBATCH --output=logs/out/R-%x/R-%x_%A.out
#SBATCH --error=logs/error/R-%x/R-%x_%A.err
#
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1            # Request 1 GPU (RTX6000 ADA)
#SBATCH --mem=64G               # RAM per job (128G max per node)
#SBATCH --cpus-per-task=8       # Adjust based on training script (or set 4–16)
#
#SBATCH --mail-user=testing.siqi@gmail.com
#SBATCH --mail-type=END,FAIL

# Correctly echo the job name (use quotes for safety)
date
echo "Slurm Job Name: ${SLURM_JOB_NAME}"
echo "Slurm Job ID: ${SLURM_JOB_ID}"

# --- User Configuration for Hugging Face ---
JUDGE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
BASE_MODEL_NAME=$(basename "${BASE_MODEL}")
GENERIC_MODEL_NAME="Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"
# GENERIC_MODEL_NAME="Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-3-small-dataset"
HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
HF_USERNAME="Siqi-Hu"
# --- End User Configuration ---

# Load modules
ml releases/2023a
ml Python
ml CUDA
srun python --version
srun nvcc --version

# Define paths
RUN_ID="run_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
SCRATCH_RUN_DIR="${GLOBALSCRATCH}/runs/${SLURM_JOB_NAME}/${RUN_ID}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR}" # Directory where sbatch was run
DATASET_DIR="${SCRATCH_RUN_DIR}/data/evaluation/sentence_completion_tasks/${GENERIC_MODEL_NAME}"

echo "Submission directory: ${SUBMIT_DIR}"
echo "Scratch run directory: ${SCRATCH_RUN_DIR}"
echo "Evaluation dataset directory: ${DATASET_DIR}"

# Create the target directory
mkdir -p "$SCRATCH_RUN_DIR"
echo "Created directory: ${SCRATCH_RUN_DIR}"

# Copy source code and data - using SUBMIT_DIR is safer
echo "Copying src directory..."
cp -r "${SUBMIT_DIR}/src/" "${SCRATCH_RUN_DIR}/"
echo "Copying data directory..."
cp -r "${SUBMIT_DIR}/data/" "${SCRATCH_RUN_DIR}/"
echo "Copying prompt_templates directory..."
cp -r "${SUBMIT_DIR}/prompt_templates/" "${SCRATCH_RUN_DIR}/"

# Change to the run directory
echo "Changing directory to ${SCRATCH_RUN_DIR}"
cd "${SCRATCH_RUN_DIR}"

# Load venv
echo "Loading python venv from ${SUBMIT_DIR}..."
source "${SUBMIT_DIR}/.venv/bin/activate"

srun echo "========================= Logging into Hugging Face ========================="
# Login using the token from the environment variable
# Using srun ensures this runs on the allocated compute node which has network access
srun huggingface-cli login --token $HF_TOKEN --add-to-git-credential
login_exit_code=$?
if [ $login_exit_code -ne 0 ]; then
    echo "Error: Hugging Face login failed with exit code $login_exit_code."
    # Optionally exit the script if login fails
    # exit 1
fi
srun echo "Hugging Face login attempted."

srun echo "=============================== Job started! ==============================="
srun python src/evaluations/judgement.py \
    --evaluation_dataset_folder "${DATASET_DIR}" \
    --judge_model_id "${JUDGE_MODEL}" \
    --output_file "evaluation_with_metrics_judement.json"

srun echo "Copy the output file to ${HOME}"
scp "${DATASET_DIR}/evaluation_with_metrics_judement.json" "${HOME}/thesis/SW-UnlearningLM/data/evaluation/sentence_completion_tasks/${GENERIC_MODEL_NAME}"

srun echo "=============================== Job completed! ==============================="

date