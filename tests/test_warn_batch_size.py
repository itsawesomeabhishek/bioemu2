"""Tests for batch-size warnings that do not change sampling."""

import logging
from unittest.mock import Mock, patch

import pytest

from bioemu2.sampling import warn_batch_size


def test_batch_size_one_warns_when_more_samples_are_requested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("bioemu2.sampling.torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING, logger="bioemu2.sampling"),
    ):
        warn_batch_size(num_tokens=200, num_samples=8, batch_size=1)

    assert "may be inefficient" in caplog.text


def test_batch_size_one_takes_priority_over_underutilization(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("bioemu2.sampling.torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING, logger="bioemu2.sampling"),
    ):
        warn_batch_size(num_tokens=64, num_samples=128, batch_size=1)

    assert "may be inefficient" in caplog.text
    assert "underutilize" not in caplog.text


def test_small_system_with_many_samples_warns_about_underuse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("bioemu2.sampling.torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING, logger="bioemu2.sampling"),
    ):
        warn_batch_size(num_tokens=64, num_samples=128, batch_size=32)

    assert "may underutilize the GPU" in caplog.text


def test_oversized_batch_warns_about_memory(caplog: pytest.LogCaptureFixture) -> None:
    device = Mock(total_memory=8 * 1024**3)
    with (
        patch("bioemu2.sampling.torch.cuda.is_available", return_value=True),
        patch("bioemu2.sampling.torch.cuda.get_device_properties", return_value=device),
        caplog.at_level(logging.WARNING, logger="bioemu2.sampling"),
    ):
        warn_batch_size(num_tokens=4000, num_samples=8, batch_size=64)

    assert "may exceed GPU memory" in caplog.text
    assert "try a batch_size of 1 or lower" in caplog.text


def test_reasonable_batch_size_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch("bioemu2.sampling.torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING, logger="bioemu2.sampling"),
    ):
        warn_batch_size(num_tokens=200, num_samples=8, batch_size=8)

    assert caplog.records == []


@pytest.mark.parametrize(
    ("num_samples", "batch_size"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_warn_batch_size_rejects_non_positive_counts(num_samples: int, batch_size: int) -> None:
    with pytest.raises(ValueError):
        warn_batch_size(num_tokens=32, num_samples=num_samples, batch_size=batch_size)
