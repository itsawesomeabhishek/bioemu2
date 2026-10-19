from typing import Any

import pytest
import torch

from bioemu2.biomoa.modules.diffusion import SingleConditioning
from bioemu2.biomoa.modules.encoders import AtomAttentionEncoder

ATOM_S = 16
ATOM_Z = 8
TOKEN_S = 12
TOKEN_Z = 8
N_ATOMS = 8
N_TOKENS = 3


def _make_encoder(*, downscale_condition_contrib: bool = False) -> AtomAttentionEncoder:
    torch.manual_seed(0)
    return AtomAttentionEncoder(
        atom_s=ATOM_S,
        atom_z=ATOM_Z,
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atoms_per_window_queries=4,
        atoms_per_window_keys=8,
        structure_prediction=True,
        use_atom_encoder=False,
        use_atom_to_token_aggregation=False,
        use_dynamics_conditioning=True,
        downscale_condition_contrib=downscale_condition_contrib,
        dt_embed_num_timescales=4,
    )


def _run_q(
    encoder: AtomAttentionEncoder,
    *,
    r: torch.Tensor,
    coords_cond: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    batch_size = dt.shape[0]
    multiplicity = r.shape[0] // batch_size
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        _, q, _, _ = encoder(
            atom_ref_pos=torch.randn(batch_size, N_ATOMS, 3, generator=generator),
            ref_charge=torch.zeros(batch_size, N_ATOMS),
            ref_element=torch.zeros(batch_size, N_ATOMS, dtype=torch.long),
            ref_atom_name_chars=torch.zeros(batch_size, N_ATOMS, 4, dtype=torch.long),
            atom_ref_space_uid=torch.zeros(batch_size, N_ATOMS, dtype=torch.long),
            atom_pad_mask=torch.ones(batch_size, N_ATOMS, dtype=torch.bool),
            atom_to_token_ids=(torch.arange(N_ATOMS) % N_TOKENS).expand(batch_size, -1),
            token_pad_mask=torch.ones(batch_size, N_TOKENS, dtype=torch.bool),
            s_trunk=torch.zeros(batch_size, N_TOKENS, TOKEN_S),
            z=torch.zeros(batch_size, N_TOKENS, N_TOKENS, TOKEN_Z),
            r=r,
            multiplicity=multiplicity,
            coords_cond=coords_cond,
            is_dynamics_conditioned=torch.ones(batch_size, 1, dtype=torch.bool),
            dt=dt,
        )
    return q


def test_lag_gate_decays_with_dt() -> None:
    encoder = _make_encoder(downscale_condition_contrib=True)
    torch.nn.init.normal_(encoder.cond_to_q_trans.weight)
    r = torch.randn(1, N_ATOMS, 3)
    coords_cond = torch.randn(1, N_ATOMS, 3)

    deltas = []
    for lag in [1e-3, 1e2, 1e6, 1e12]:
        q = _run_q(
            encoder,
            r=r,
            coords_cond=coords_cond,
            dt=torch.tensor([[lag]]),
        )
        q_unconditioned = _run_q(
            encoder,
            r=r,
            coords_cond=torch.zeros_like(coords_cond),
            dt=torch.tensor([[lag]]),
        )
        deltas.append((q - q_unconditioned).abs().max().item())

    ungated = torch.nn.functional.linear(coords_cond, encoder.cond_to_q_trans.weight)
    assert deltas[0] == pytest.approx(ungated.abs().max().item(), rel=1e-3)
    assert all(later < earlier for earlier, later in zip(deltas, deltas[1:]))
    assert deltas[-1] == pytest.approx(0.0, abs=1e-6)


def test_single_conditioner_injects_masked_dt() -> None:
    conditioner = SingleConditioning(
        sigma_data=16.0,
        token_s=TOKEN_S,
        s_input_dim=TOKEN_S,
        a_s=TOKEN_S,
        dim_fourier=8,
        num_transitions=1,
        use_dt_conditioning=True,
        dt_embed_num_timescales=4,
    )
    kwargs: dict[str, Any] = {
        "times": torch.zeros(2),
        "s_trunk": torch.randn(2, N_TOKENS, TOKEN_S),
        "s_inputs": torch.randn(2, N_TOKENS, TOKEN_S),
    }
    dt = torch.tensor([[10.0], [20.0]])
    with torch.no_grad():
        torch.nn.init.normal_(conditioner.dt_embed_to_single.weight, std=0.1)
        masked, _ = conditioner(
            dt=dt,
            is_dynamics_conditioned=torch.zeros(2, 1, dtype=torch.bool),
            **kwargs,
        )
        conditioned, _ = conditioner(
            dt=dt,
            is_dynamics_conditioned=torch.ones(2, 1, dtype=torch.bool),
            **kwargs,
        )
        infinite_lag, _ = conditioner(
            dt=torch.full((2, 1), 1e30),
            is_dynamics_conditioned=torch.ones(2, 1, dtype=torch.bool),
            **kwargs,
        )

    assert not torch.allclose(conditioned, masked)
    torch.testing.assert_close(infinite_lag, masked)
