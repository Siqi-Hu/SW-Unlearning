#!/bin/bash
#SBATCH --job-name=test_parallelism 
#SBATCH --output=logs/out/R-%x_%A.out
#SBATCH --error=logs/error/R-%x_%A.err
#
#SBATCH --nodes=2               # Request 2 nodes (1 GPU each)
#SBATCH --ntasks-per-node=1     # One task per node
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1            # Request 1 GPU (RTX6000 ADA) per node
#SBATCH --mem=128G              # RAM per job (128G max per node)
#SBATCH --cpus-per-task=8       # Adjust based on training script (or set 4–16)
#
#SBATCH --mail-user=testing.siqi@gmail.com
#SBATCH --mail-type=END,FAIL


# Correctly echo the job name (use quotes for safety)
date
echo "Slurm Job Name: ${SLURM_JOB_NAME}"
echo "Slurm Job ID: ${SLURM_JOB_ID}"
srun nvidia-smi  # check GPU memory usage

date
