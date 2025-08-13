#!/bin/bash
#SBATCH --job-name=gen_prompts
#SBATCH --output=result/out/R-%x_%A_%a_%j.out
#SBATCH --error=result/error/R-%x_%A_%a_%j.err
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --array=0-19
#SBATCH --mem-per-cpu=200000
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

srun echo "=============================== Job started! ==============================="

srun python src/evaluations/generate_star_wars_prompts.py --start=$((SLURM_ARRAY_TASK_ID*5)) --num_prompts=5

srun echo "=============================== Job completed! ==============================="

date