import random

import numpy as np
import torch


def seed_random_generators(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch global random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
