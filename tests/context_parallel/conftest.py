from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel

from bioemu2.biomoa.models.score_model import BioMoaCModel
from tests.context_parallel.helpers import randomize_params


def _build_score_model(config: DictConfig) -> BioMoaCModel:
    torch.manual_seed(0)
    model = instantiate(config).eval()
    randomize_params(model)
    return model


@pytest.fixture
def score_model_config() -> DictConfig:
    """Tiny BioMoaC config shared by CP model, API and CLI tests."""
    config = OmegaConf.load(Path(__file__).parents[1] / "assets/tiny_config.yaml")
    config.token_z = config.sequence_embedder.token_z = 32
    return config


@pytest.fixture
def score_model(score_model_config: DictConfig) -> BioMoaCModel:
    """Deterministic tiny model whose zero-initialized parameters are randomized."""
    return _build_score_model(score_model_config)


@pytest.fixture
def method_pair_score_model(score_model_config: DictConfig) -> BioMoaCModel:
    """Tiny model that also embeds method conditioning into pairs."""
    config = score_model_config.copy()
    config.sequence_embedder.embed_method_into_pairs = True
    return _build_score_model(config)


@pytest.fixture
def checkpoint(tmp_path: Path, score_model_config: DictConfig, score_model: BioMoaCModel) -> Path:
    """Save the tiny model as an EMA checkpoint next to its ``config.yaml``."""
    OmegaConf.save({"score_model": score_model_config}, tmp_path / "config.yaml")
    _checkpoint = tmp_path / "checkpoint.ckpt"
    torch.save({"model_ema": AveragedModel(score_model).state_dict()}, _checkpoint)
    return _checkpoint
