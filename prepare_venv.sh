#!/bin/bash

ml releases/2023a
ml Python
ml CUDA

# Check if .venv exists and remove it if necessary
echo "Checking for existing virtual environment..."
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create and activate a virtual environment
echo "Creating and activating virtual environment..."
python -m venv .venv
source .venv/bin/activate
pip install ruff

# Install project dependencies
echo "Installing project dependencies..."
# pip install -r requirements.txt --index-strategy unsafe-best-match
pip install -r requirements.txt

# Install additional tools
echo "Installing additional tools..."
pip install pip-system-certs pre-commit

echo "Setup complete."