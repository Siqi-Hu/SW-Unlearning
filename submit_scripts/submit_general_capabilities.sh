#!/bin/bash
#SBATCH --job-name=general-capabilities
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

date
ml releases/2023a
ml Python
ml CUDA
srun python --version
srun nvcc --version
source ./.venv/bin/activate

srun nvidia-smi  # check GPU memory usage


srun echo "=============================== Job started! ==============================="

srun python src/evaluations/general_capabilities.py

srun echo "=============================== Job completed! ==============================="

date
