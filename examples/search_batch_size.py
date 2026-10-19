# %% [markdown]
# # Find a sampling batch size
#
# Edit the setup and sampling call below, then run from the repository root:
# `CUDA_VISIBLE_DEVICES=0 python examples/search_batch_size.py`.
# Times warmed sampling with CUDA events, not model loading, embedding, filtering, or file I/O.
# Use an idle GPU; the search may deliberately encounter CUDA OOMs.

# %%
import gc
import logging
from collections.abc import Callable
from statistics import median

import torch

from bioemu2 import (
    EDMSampler,
    System,
    get_msa,
    load_model,
    sample_with_embeddings,
)
from bioemu2.paths import EXAMPLES_DIR

logger = logging.getLogger(__name__)


def measure_batch(sample: Callable[[int], object], size: int, device: torch.device) -> float:
    """Log median CUDA-event batch time and return samples/sec; zero means CUDA OOM."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        sample(size)  # Untimed warmup; discard every output before the next call.
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        times = []
        for _ in range(3):
            start.record(stream)
            sample(size)
            end.record(stream)
            end.synchronize()
            times.append(start.elapsed_time(end) / 1000)
        seconds = median(times)
        logger.info("batch=%d time=%.3fs speed=%.3f samples/s", size, seconds, size / seconds)
        return size / seconds
    except torch.cuda.OutOfMemoryError:
        logger.info("batch=%d CUDA OOM", size)
    # Release the failed call's traceback before clearing cached allocations.
    gc.collect()
    torch.cuda.empty_cache()
    return 0.0


def search_batch_sizes(
    measure: Callable[[int], float], max_batch_size: int = 128, min_improvement: float = 0.01
) -> dict[int, float]:
    """Return measured rates, doubling then refining near the best; ties prefer smaller sizes."""
    if max_batch_size < 1 or not 0 <= min_improvement < 1:
        raise ValueError("Require max_batch_size >= 1 and 0 <= min_improvement < 1.")
    rates = {}

    def trial(size: int) -> float:
        rates[size] = measure(size)
        return rates[size]

    previous_rate = trial(1)
    if not previous_rate:
        raise RuntimeError("Batch size 1 ran out of CUDA memory.")
    size, stalls = 1, 0
    while size < max_batch_size:
        next_size = min(2 * size, max_batch_size)
        rate = trial(next_size)
        if not rate:
            low, high = size, next_size
            while high - low > 1:
                middle = (low + high) // 2
                if trial(middle):
                    low = middle
                else:
                    high = middle
            break
        stalls = stalls + 1 if rate < previous_rate * (1 + min_improvement) else 0
        previous_rate = rate
        size = next_size
        if stalls >= 2:
            break

    while True:
        sizes = sorted(size for size in rates if rates[size])
        best = max(sizes, key=rates.__getitem__)
        index = sizes.index(best)
        neighbors = sizes[max(0, index - 1) : index] + sizes[index + 1 : index + 2]
        candidates = [
            (best + neighbor) // 2
            for neighbor in neighbors
            if abs(best - neighbor) > 1
            and rates[neighbor] < rates[best] / (1 + min_improvement)
            and (best + neighbor) // 2 not in rates
        ]
        if not candidates:
            return rates
        for size in candidates:
            trial(size)


# %% Edit this setup and sampling call for your workload.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("highest")
    system = System.from_yaml(EXAMPLES_DIR / "systems/ake.yaml")
    msa = get_msa(system)
    model = load_model("bioemu2-preview", device="cuda", precision="32-true")
    embeddings = model.embed(system, msa=msa, mode="equilibrium")
    sampler = EDMSampler(num_steps=200, step_scale=1.0)

    def sample(batch_size: int) -> object:
        return sample_with_embeddings(
            system,
            embeddings,
            model=model,
            num_samples=batch_size,
            batch_size=batch_size,
            sampler=sampler,
            seed=42,
            progress=False,
        )

    rates = search_batch_sizes(
        lambda size: measure_batch(sample, size, model.device), max_batch_size=128
    )
    best = max(sorted(rates), key=rates.__getitem__)
    logger.info("Best measured batch=%d speed=%.3f samples/s", best, rates[best])
