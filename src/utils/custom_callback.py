import json
import os

from huggingface_hub import upload_folder
from peft import PeftModel
from transformers import TrainerCallback


# Custom callback to save metrics
class MetricsSavingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics = {
            "train_loss": [],
            "train_steps": [],
            "train_epochs": [],
            "eval_loss": [],
            "eval_steps": [],
            "eval_epochs": [],
        }
        self.initial_eval_done = False
        os.makedirs(output_dir, exist_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        """Capture initial evaluation loss at step 0"""
        print("Performing initial evaluation at step 0...")

        # Get the trainer from kwargs
        if "model" in kwargs and hasattr(kwargs.get("eval_dataset"), "__len__"):
            try:
                # Manual initial evaluation
                trainer = kwargs.get("trainer")  # This might not always be available
                if trainer:
                    # Temporarily set to evaluation mode
                    initial_metrics = trainer.evaluate()
                    if "eval_loss" in initial_metrics:
                        self.metrics["eval_loss"].append(initial_metrics["eval_loss"])
                        self.metrics["eval_steps"].append(0)
                        self.metrics["eval_epochs"].append(0.0)
                        self.initial_eval_done = True
                        print(
                            f"Initial eval loss at step 0: {initial_metrics['eval_loss']:.4f}"
                        )
            except Exception as e:
                print(f"Could not perform initial evaluation: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            epoch = state.epoch

            # save training loss
            if "loss" in logs:
                self.metrics["train_loss"].append(logs["loss"])
                self.metrics["train_steps"].append(step)
                self.metrics["train_epochs"].append(epoch)

            if "eval_loss" in logs:
                self.metrics["eval_loss"].append(logs["eval_loss"])
                self.metrics["eval_steps"].append(step)
                self.metrics["eval_epochs"].append(epoch)

            with open(f"{self.output_dir}/training_metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)


class PushLoRACheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        adapter_subdir = f"lora-checkpoint-{step}"
        output_path = os.path.join(args.output_dir, adapter_subdir)
        model = kwargs["model"]

        # Save only the LoRA adapter
        if isinstance(model, PeftModel):
            model.save_pretrained(output_path, safe_serialization=True)

        # Push to hub under a unique revision (or use branch=... to isolate)
        upload_folder(
            repo_id=args.hub_model_id,
            folder_path=output_path,
            path_in_repo=adapter_subdir,
            commit_message=f"Pushed LoRA adapter at step {step}",
        )
