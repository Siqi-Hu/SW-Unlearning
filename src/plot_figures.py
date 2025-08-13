from utils.visualizer import plot_loss_over_steps

path = "/home/ucl/ingi/sihu/thesis/SW-UnlearningLM/plots/Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"

finetuned_metrics_file = f"{path}/training_metrics.json"
plot_loss_over_steps(finetuned_metrics_file)
