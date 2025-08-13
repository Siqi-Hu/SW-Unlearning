#!/bin/bash
#SBATCH --job-name=submit_test
#SBATCH --output=logs/out/R-%x_%A.out
#SBATCH --error=logs/error/R-%x_%A.err
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

echo "Submission directory: ${SUBMIT_DIR}"
echo "Scratch run directory: ${SCRATCH_RUN_DIR}"

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

# Print info
echo "Current directory: $(pwd)"
echo "Using pip: $(which pip)"
ls -l -a