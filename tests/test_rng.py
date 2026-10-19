import random

import numpy as np
import torch

from bioemu2.rng import seed_random_generators


def test_seed_random_generators_reproducibly_seeds_all_generators() -> None:
    seed_random_generators(42)
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)

    seed_random_generators(42)

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)
