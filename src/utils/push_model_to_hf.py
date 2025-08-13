import argparse
import logging
import os
import sys

from huggingface_hub import HfApi, create_repo, login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Ensure logs go to SLURM output
)


def main():
    parser = argparse.ArgumentParser(
        description="Create a Hugging Face repo and upload model files."
    )
    parser.add_argument(
        "--local_model_dir",
        type=str,
        required=True,
        help="Path to the local directory containing the model files.",
    )
    parser.add_argument("--repo_type", type=str, required=True, default="model")
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="The desired Hugging Face repository ID (e.g., 'username/repo-name').",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=False,
        default=None,
        help="Hugging Face API token (optional, attempts login/env var if not provided).",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fine-tuned model",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository."
    )  # Optional: add flag for private repos

    args = parser.parse_args()

    # --- Argument Validation ---
    if not os.path.isdir(args.local_model_dir):
        logging.error(f"Local model directory not found: {args.local_model_dir}")
        sys.exit(1)

    # --- Authentication ---
    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        logging.warning(
            "HF token not provided via argument or HF_TOKEN env var. Attempting interactive login or cached credentials."
        )
        # Rely on cached credentials or potential interactive login if run manually
        # Note: Interactive login won't work in a non-interactive SLURM job.
        # For SLURM, ensure HF_TOKEN is set or passed via --hf_token.
        # Optionally, explicitly call login() here if needed, but it's often automatic.
        # login(token=token) # Usually not needed if token is passed to HfApi/create_repo
    else:
        logging.info("Using provided Hugging Face token.")

    # --- Create Repository ---
    try:
        logging.info(f"Ensuring repository '{args.hf_repo_id}' exists...")
        repo_url = create_repo(
            args.hf_repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
            token=token,  # Pass token for authentication
        )
        logging.info(f"Repository '{args.hf_repo_id}' ensured. URL: {repo_url}")

    except Exception as e:
        logging.error(f"Failed to create or access repository '{args.hf_repo_id}': {e}")
        sys.exit(1)

    # --- Upload Files ---
    try:
        logging.info(
            f"Uploading contents from '{args.local_model_dir}' to '{args.hf_repo_id}'..."
        )
        api = HfApi(token=token)  # Pass token for authentication
        api.upload_folder(
            folder_path=args.local_model_dir,
            repo_id=args.hf_repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            # ignore_patterns=['*.log', 'checkpoint-*'] # Example ignore patterns
        )
        logging.info(f"Successfully uploaded model files to '{args.hf_repo_id}'.")

    except Exception as e:
        logging.error(f"An error occurred during model upload: {e}")
        sys.exit(1)

    logging.info("Script finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
