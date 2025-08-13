#!/bin/bash
#SBATCH --job-name=prepare_dataset_llama_seq
#SBATCH --output=logs/out/R-%x_%A.out
#SBATCH --error=logs/error/R-%x_%A.err
#
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
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
BOOTSTRAP_COEF=5
BASE_MODEL_ID="meta-llama/Meta-Llama-3-8B"
BASE_MODEL_NAME=$(basename "${BASE_MODEL_ID}")
REINFORCED_MODEL_ID="Siqi-Hu/${BASE_MODEL_NAME}-lora-starwars-finetuned-epoch-10"
HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
HF_USERNAME="Siqi-Hu"
HF_REPO_NAME="${BASE_MODEL_NAME}-generic-predictions-starwars"
HF_REPO_ID="${HF_USERNAME}/${HF_REPO_NAME}-bootstrap-coef-${BOOTSTRAP_COEF}-seq"
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
OUTPUT_DIR="${SCRATCH_RUN_DIR}/data/generic_predictions/${HF_REPO_NAME}"

echo "Submission directory: ${SUBMIT_DIR}"
echo "Scratch run directory: ${SCRATCH_RUN_DIR}"
echo "Dataset output directory: ${OUTPUT_DIR}"


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

srun echo "=============================== Job started! ==============================="

srun python src/prepare_dataset_seq.py \
    --context_length 128 \
    --base_model_id "${BASE_MODEL_ID}" \
    --reinforced_model_id "${REINFORCED_MODEL_ID}" \
    --dict_file ./data/dictionary/consolidated_dictionary.json \
    --input_file_dir ./data/star_wars_transcripts/ \
    --output_file "${OUTPUT_DIR}/generic_predictions.hf" \
    --bootstrap_coef $BOOTSTRAP_COEF \
    --hf_repo_id "${HF_REPO_ID}"

# srun echo "================== Preparing to Push to Hugging Face Hub =================="

# srun python src/utils/push_model_to_hf.py \
#     --local_model_dir "${OUTPUT_DIR}/generic_predictions.hf" \
#     --repo_type "dataset" \
#     --hf_repo_id "${HF_REPO_ID}" \
#     --commit_message "Upload from job ${SLURM_JOB_NAME}-${SLURM_JOB_ID}" 

srun echo "=============================== Job completed! ==============================="

date