"""Sample protein structures across multiple GPUs.

Launch with torchrun as described in docs/context-parallel-sampling.md.
For installation and hardware requirements, see the notebook
quickstart in README.md.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import py3Dmol
import torch.distributed as dist
from playwright.sync_api import sync_playwright

from bioemu2 import System, get_msa
from bioemu2.context_parallel import api as cp
from bioemu2.paths import EXAMPLES_DIR
from bioemu2.visualization import view_structure_grid

RENDER_WIDTH = 1400
RENDER_HEIGHT = 700


def save_view_png(
    view: py3Dmol.view,
    output_path: Path,
    *,
    width: int,
    height: int,
) -> None:
    with TemporaryDirectory() as directory:
        html_path = Path(directory) / "viewer.html"
        view.write_html(str(html_path))
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=2,
            )
            page.goto(html_path.as_uri(), wait_until="networkidle")
            page.locator("canvas").first.wait_for()
            page.wait_for_timeout(2000)
            page.screenshot(path=str(output_path), full_page=True)
            browser.close()

start_time = perf_counter()
with cp.session():
    system = System.from_yaml(EXAMPLES_DIR / "systems/myoglobin_101m.yaml")
    msa = get_msa(system) if dist.get_rank() == 0 else None
    model = cp.load_model("bioemu2-preview", precision="bf16-mixed")
    output_dir = Path("outputs/myoglobin-cp")
    samples = cp.sample_equilibrium(
        system,
        model=model,
        msa=msa,
        num_samples=8,
        sampler="edm",
        seed=42,
        output_dir=output_dir,
    )
    if dist.get_rank() == 0:
        view = view_structure_grid(
            samples,
            rows=2,
            columns=4,
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
        )
        save_view_png(
            view,
            output_dir / "samples.png",
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
        )
        print(f"Completed in {perf_counter() - start_time:.1f} seconds.")
