#!/bin/bash
#SBATCH --job-name=gen_dict
#SBATCH --output=result/out/R-%x_%A_%a_%j.out
#SBATCH --error=result/error/R-%x_%A_%a_%j.err
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --array=0-208
#SBATCH --mem-per-cpu=300000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
#SBATCH --mail-user=testing.siqi@gmail.com
#SBATCH --mail-type=END,FAIL


date
ml releases/2023a
ml Python
ml CUDA
srun python --version
srun nvcc --version
#
#srun python --version
#python -m venv .venv
source .venv/bin/activate

#pip install -r requirements.txt

#srun pip list

srun nvidia-smi  # check GPU memory usage

srun echo "=============================== Job started! ==============================="

srun python src/generate_dict/generate_dict.py --start=$((SLURM_ARRAY_TASK_ID*10)) --end=$(((SLURM_ARRAY_TASK_ID+1)*10))

srun echo "=============================== Job completed! ==============================="

date
