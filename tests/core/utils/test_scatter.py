# NOTE: Adapted from `mdmlff.tests.test_mlff.test_scatter`.
import torch

from bioemu2.structure.scatter import scatter, scatter_mean, scatter_sum

SRC_TENSOR = torch.randn(6)
INDEX = torch.tensor([0, 1, 0, 1, 2, 1])
OUT_TENSOR = torch.zeros(3)
DIM = 0


def test_scatter_sum() -> None:
    """Test for pure torch scatter sum reduction."""
    actual = scatter(src=SRC_TENSOR, index=INDEX, dim=DIM, reduce="sum")
    actual_wrapped = scatter_sum(src=SRC_TENSOR, index=INDEX, dim=DIM)
    expected = torch.zeros_like(OUT_TENSOR).scatter_reduce_(
        dim=DIM, index=INDEX, src=SRC_TENSOR, reduce="sum", include_self=False
    )
    assert torch.equal(actual, expected)
    assert torch.equal(actual_wrapped, expected)


def test_scatter_mean() -> None:
    """Test for pure torch scatter mean reduction."""
    actual = scatter(src=SRC_TENSOR, index=INDEX, dim=DIM, reduce="mean")
    actual_wrapped = scatter_mean(src=SRC_TENSOR, index=INDEX, dim=DIM)
    expected = OUT_TENSOR.scatter_reduce_(
        dim=DIM, index=INDEX, src=SRC_TENSOR, reduce="mean", include_self=False
    )
    assert torch.equal(actual, expected)
    assert torch.equal(actual_wrapped, expected)
