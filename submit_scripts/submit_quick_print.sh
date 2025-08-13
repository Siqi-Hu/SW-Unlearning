#!/bin/bash
#SBATCH --job-name=submit_quick_print
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


date
ml releases/2023a
ml Python
ml CUDA
srun python --version
srun nvcc --version
source ./.venv/bin/activate

srun nvidia-smi  # check GPU memory usage


srun echo "=============================== Job started! ==============================="

# srun python src/evaluations/complete_sentences.py \
#     --base_model_id "meta-llama/Llama-2-7b-hf" \
#     --reinforced_model_id "Siqi-Hu/Llama2-7B-lora-r-32-finetuned-epoch-4" \
#     --generic_model_id "Siqi-Hu/Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0" \
#     --evaluation_dataset_folder "/home/ucl/ingi/sihu/thesis/SW-UnlearningLM/data/evaluation/llm-as-a-judge/" 

srun python src/experiments/next_token_prob_over_steps.py

srun echo "=============================== Job completed! ==============================="

date
