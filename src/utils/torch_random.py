import random

import numpy as np
import torch


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)  # For Python's built-in random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic results on GPUs
    torch.backends.cudnn.benchmark = (
        False  # Disables optimization that could introduce non-determinism
    )
