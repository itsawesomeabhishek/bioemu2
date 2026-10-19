# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Context-parallel model classes and factory functions."""

import torch.nn as nn
from torch.optim.swa_utils import AveragedModel

from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.biomoa.modules.pairmixer import (
    PairMixerSequenceEmbedder as BioMoaPairMixerSequenceEmbedder,
)
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.models.score_model import (
    BioMoaCDistributed,
    CPTrunkEmbeddings,
)
from bioemu2.context_parallel.modules.pairmixer import (
    PairMixerSequenceEmbedderDistributed,
)

__all__ = [
    "BioMoaCDistributed",
    "CPTrunkEmbeddings",
    "PairMixerSequenceEmbedderDistributed",
    "wrap_score_model_for_cp",
]


def wrap_score_model_for_cp(score_model: nn.Module, dist_manager: DistributedManager) -> nn.Module:
    """Wrap a fully loaded serial ``score_model`` with its distributed CP analogue.

    The serial instance is consumed by the distributed constructor; weights
    are reused, not copied.
    """
    inner = score_model.module if isinstance(score_model, AveragedModel) else score_model

    seq_embedder = getattr(inner, "sequence_embedder", None)
    if isinstance(inner, BioMoaCModel) and isinstance(
        seq_embedder, BioMoaPairMixerSequenceEmbedder
    ):
        return BioMoaCDistributed(
            inner,
            dist_manager,
            sequence_embedder=PairMixerSequenceEmbedderDistributed(seq_embedder, dist_manager),
        ).eval()
    raise NotImplementedError(
        "Context-parallel sampling supports BioMoaCModel with the BioMoa "
        "PairMixerSequenceEmbedder only; got "
        f"{type(inner).__name__} with {type(seq_embedder).__name__}."
    )
