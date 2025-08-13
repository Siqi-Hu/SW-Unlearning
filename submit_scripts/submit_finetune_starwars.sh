#!/bin/bash
#SBATCH --job-name=finetune_starwars_llama_all_linear
#SBATCH --output=logs/out/R-%x/R-%x_%A.out
#SBATCH --error=logs/error/R-%x/R-%x_%A.err
#
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
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
BASE_MODEL="meta-llama/Meta-Llama-3-8B"
# BASE_MODEL="meta-llama/Llama-2-7b-hf"
BASE_MODEL_NAME=$(basename "${BASE_MODEL}")
HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
HF_USERNAME="Siqi-Hu"
# HF_REPO_NAME="Llama2-7B-lora-r-32-finetuned-epoch-4"
HF_REPO_NAME="Llama3-8B-lora-r-32-finetuned-epoch-3"
HF_REPO_ID="${HF_USERNAME}/${HF_REPO_NAME}"
# --- End User Configuration ---

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Please set it before submitting the job: export HF_TOKEN='hf_...'"
  exit 1
fi
echo "Hugging Face Username: ${HF_USERNAME}"
echo "Target Hugging Face Repo ID: ${HF_REPO_ID}"


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
OUTPUT_DIR="${SCRATCH_RUN_DIR}/models/${HF_REPO_NAME}"

echo "Submission directory: ${SUBMIT_DIR}"
echo "Scratch run directory: ${SCRATCH_RUN_DIR}"
echo "Model output directory: ${OUTPUT_DIR}"

# Create the target directory
mkdir -p "$SCRATCH_RUN_DIR"
echo "Created directory: ${SCRATCH_RUN_DIR}"

# Copy source code and data - using SUBMIT_DIR is safer
echo "Copying src directory..."
cp -r "${SUBMIT_DIR}/src/" "${SCRATCH_RUN_DIR}/"
echo "Copying data directory..."
cp -r "${SUBMIT_DIR}/data/" "${SCRATCH_RUN_DIR}/"

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

srun echo "========================= Starting Fine-tuning Job ========================="

srun python src/finetune_model_starwars.py \
    --model_name "${BASE_MODEL}" \
    --input_file_dir ./data/star_wars_transcripts/ \
    --output_dir "${OUTPUT_DIR}" \
    --context_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --hub_model_id "${HF_REPO_NAME}"

srun echo "============================== Job completed! =============================="