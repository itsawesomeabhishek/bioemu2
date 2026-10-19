# Development

Use [uv](https://docs.astral.sh/uv/) to install the same dependency versions
used in tests:

```bash
uv sync --extra cuda
```

On GPUs with compute capability 7.0 (Volta), use the CUDA 12.6 build instead. Pass the same
extra to every `uv run`, because `uv run` otherwise reinstalls the default `cuda`
build:

```bash
uv sync --extra cuda12
uv run --extra cuda12 pytest tests
uv run --extra cuda12 bioemu2 sample-equilibrium \
  --pdb-id 1AKE \
  --out outputs/adk
```

GPUs with compute capability 7.0 (Volta) have no native bfloat16 support; with
`bf16-mixed` precision, PyTorch autocast emulates bfloat16 on these GPUs.

Run tests with:

```bash
uv run pytest tests
```

Run other commands with `uv run`. For notebooks, install the visualization
dependencies and register a kernel:

```bash
uv sync --extra cuda --extra viz
uv run python -m ipykernel install --user \
  --name bioemu2 --display-name "BioEmu2"
```
