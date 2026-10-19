# Load a model checkpoint

`load_model()` returns a model you pass to the sampling functions. By default it
downloads the `bioemu2-preview` checkpoint from the
[`microsoft/bioemu2`](https://huggingface.co/microsoft/bioemu2) Hugging Face
repository:

```python
from bioemu2 import load_model

model = load_model(
    device="cuda",
    precision="bf16-mixed",
)
```

If `device` is omitted, BioEmu2 uses CUDA when available and CPU otherwise.
`load_model()` defaults to the fast settings that work well across the sampling
workflows: `"bf16-mixed"` precision for the trunk embedding and `"high"` float32
matmul precision for the denoiser. Pass `precision="32-true"` and
`matmul_precision=None` to run both in full precision.

## Choose where checkpoints are cached

The checkpoint and its configuration are downloaded and cached automatically in
the standard Hugging Face cache; the log reports where the checkpoint is stored.
To keep checkpoints elsewhere, for example on a larger disk, pass `cache_dir`:

```python
from bioemu2 import load_model

model = load_model(cache_dir="/data/bioemu2-checkpoints")
```

On the command line, use `--checkpoint-cache-dir DIR`. Setting `HF_HUB_CACHE` or
`HF_HOME` also relocates the default cache.

## Load a local checkpoint

Pass `checkpoint_path` to load a checkpoint from disk:

```python
from pathlib import Path

from bioemu2 import load_model

model = load_model(
    checkpoint_path=Path("checkpoints/model.ckpt"),
    device="cuda",
)
```

The checkpoint's `config.yaml` must be in the same directory. Pass at most one
of `checkpoint` and `checkpoint_path` to `load_model()`. For a named checkpoint,
pass `checkpoint="bioemu2-preview"`. On the command line, use
`--checkpoint-path /path/to/model.ckpt`.

Pass the loaded model to `EmuSession` for sampling and propagation:

```python
from bioemu2 import EmuSession, System

system = System.from_pdb_id("1AKE")
emu = EmuSession(system, model=model, mode="equilibrium")
samples = emu.sample(num_samples=8)
```

Or pass it to the standalone `sample_equilibrium()` function:

```python
from bioemu2 import sample_equilibrium

samples = sample_equilibrium(system, model=model, num_samples=8)
```

For a time-ordered rollout, pass the model to
[`simulate_trajectory()`](propagation.md#build-a-time-ordered-trajectory).
See the [functional sampling API](functional-api.md#sample-structures) and
[functional propagation API](functional-api.md#propagate-structures) for more
examples.
