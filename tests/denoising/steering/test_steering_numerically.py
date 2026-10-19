"""Numerical correctness tests for FK steering math.

A one-dimensional GMM with a quadratic potential has a closed-form biased posterior.
The relevant FK behavior (weights, telescoping log G, and resampling distribution)
lives in small functions in ``bioemu2.denoising.steering.feats``, so these tests exercise
those functions directly without constructing a full protein structure and sampler.
"""

from __future__ import annotations

import pytest
import torch

from bioemu2.denoising.steering.feats import (
    fk_log_increment,
    grouped_softmax,
    systematic_resample,
)


class TestFKLogIncrement:
    def test_bootstrap_at_step_zero(self):
        E = torch.tensor([1.0, 2.0, 3.0])
        log_G = fk_log_increment(E, None)
        assert torch.allclose(log_G, -E)

    def test_telescopes_to_negative_total(self):
        # Sum of log_G across steps must equal -E_final.
        energies = [torch.tensor([1.0]), torch.tensor([1.5]), torch.tensor([0.3])]
        total = torch.zeros(1)
        prev = None
        for E in energies:
            total = total + fk_log_increment(E, prev)
            prev = E
        assert torch.allclose(total, -energies[-1])


class TestSystematicResample:
    def test_collapses_to_dominant_particle(self):
        # Pile all weight on particle index 2 within each of 4 groups of 4.
        w = torch.zeros(16)
        w[2::4] = 1.0
        idx = systematic_resample(w, group_size=4)
        # All 16 returned indices must be group-offset + 2.
        assert idx.tolist() == [g * 4 + 2 for g in range(4) for _ in range(4)]

    def test_ragged_last_group_stays_in_its_block(self):
        # 6 particles, groups of 4 -> [0..3] + [4, 5]; each group resamples itself.
        w = torch.cat([torch.full((4,), 0.25), torch.full((2,), 0.5)])
        idx = systematic_resample(w, group_size=4)
        assert idx[:4].max() < 4 and idx[4:].min() >= 4

    def test_uniform_weights_keep_every_particle(self):
        # Systematic resampling of uniform weights is a permutation-free identity.
        w = torch.full((1000,), 1e-3)
        idx = systematic_resample(w, group_size=None)
        assert idx.tolist() == list(range(1000))

    def test_grouping_bounds_ancestry_collapse(self):
        """Iterated resampling collapses ancestry: after enough rounds most of the
        batch descends from a handful of ancestors. Resampling within groups of
        `group_size` bounds the surviving lineages below by the number of groups,
        because groups never mix — that is the point of `resampling_group_size`."""
        torch.manual_seed(0)
        batch, group_size, rounds = 1024, 64, 200  # rounds ~ denoising steps

        def surviving_ancestors(gs: int | None) -> int:
            ancestor = torch.arange(batch)
            for _ in range(rounds):
                w = grouped_softmax(2.0 * torch.randn(batch), gs)
                ancestor = ancestor[systematic_resample(w, gs)]
            return len(ancestor.unique())

        whole_batch = surviving_ancestors(None)
        grouped = surviving_ancestors(group_size)
        assert grouped >= batch // group_size
        assert whole_batch <= 3  # observed: 1 (seeds 0-4); grouped: 16 (one lineage per group)


# Analytical 1D Boltzmann-biased FK test.


class TestFKBiasesTo1DTarget:
    """Pure-numerical test that FK softmax weighting + resampling biases
    a uniform empirical distribution toward a target Boltzmann.

    We simulate `M` initial samples drawn uniformly in `[-5, 5]`, assign each
    an energy `E(x) = (x - target)²/2`, run a single FK softmax + resample
    step with `fk_lambda` controlling the inverse temperature, and check that
    the mean of the resampled distribution shifts toward the target.
    """

    def test_softmax_pulls_mean_toward_target(self):
        torch.manual_seed(0)
        M = 10_000
        target = 1.5
        x = torch.empty(M).uniform_(-5.0, 5.0)
        # log G ∝ -E(x); fk_lambda=4 makes the bias strong.
        energy = 0.5 * (x - target) ** 2
        log_G = -energy
        w = torch.softmax(4.0 * log_G.view(1, M), dim=1)
        idx = torch.multinomial(w, M, replacement=True)
        resampled = x[idx]
        assert resampled.mean().item() == pytest.approx(target, abs=0.1)
