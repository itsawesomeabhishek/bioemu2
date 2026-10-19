from typing import Any

import torch

from bioemu2.biomoa.modules.encoders import AtomAttentionEncoder

ATOM_S = 16
ATOM_Z = 8
TOKEN_S = 12
TOKEN_Z = 8
ATOMS_PER_WINDOW_QUERIES = 4
ATOMS_PER_WINDOW_KEYS = 8
N_ATOMS = 8
N_TOKENS = 3


def _make_conditioning_case(
    batch_size: int = 1, multiplicity: int = 1
) -> tuple[AtomAttentionEncoder, dict[str, Any], torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    encoder = AtomAttentionEncoder(
        atom_s=ATOM_S,
        atom_z=ATOM_Z,
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atoms_per_window_queries=ATOMS_PER_WINDOW_QUERIES,
        atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
        structure_prediction=True,
        use_atom_encoder=False,
        use_atom_to_token_aggregation=False,
        use_dynamics_conditioning=True,
    )
    torch.nn.init.normal_(encoder.cond_to_q_trans.weight)
    g = torch.Generator().manual_seed(123)
    tok_idx = torch.arange(N_ATOMS) % N_TOKENS
    inputs = dict(
        atom_ref_pos=torch.randn(batch_size, N_ATOMS, 3, generator=g),
        ref_charge=torch.zeros(batch_size, N_ATOMS),
        ref_element=torch.zeros(batch_size, N_ATOMS, dtype=torch.long),
        ref_atom_name_chars=torch.zeros(batch_size, N_ATOMS, 4, dtype=torch.long),
        atom_ref_space_uid=torch.zeros(batch_size, N_ATOMS),
        atom_pad_mask=torch.ones(batch_size, N_ATOMS),
        atom_to_token_ids=tok_idx.unsqueeze(0).repeat(batch_size, 1),
        token_pad_mask=torch.ones(batch_size, N_TOKENS),
        s_trunk=torch.zeros(batch_size, N_TOKENS, TOKEN_S),
        z=torch.zeros(batch_size, N_TOKENS, N_TOKENS, TOKEN_Z),
        multiplicity=multiplicity,
    )
    num_rows = batch_size * multiplicity
    r = torch.randn(num_rows, N_ATOMS, 3, generator=g)
    coords_cond = torch.randn(num_rows, N_ATOMS, 3, generator=g)
    return encoder, inputs, r, coords_cond


def _run_with_dynamics_condition(
    encoder: AtomAttentionEncoder,
    inputs: dict[str, Any],
    r: torch.Tensor,
    coords_cond: torch.Tensor,
    is_dynamics_conditioned: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        _, q, _, _ = encoder(
            **inputs,
            r=r,
            coords_cond=coords_cond,
            is_dynamics_conditioned=is_dynamics_conditioned,
        )
    return q


def test_is_dynamics_conditioned_controls_each_multiplicity_copy() -> None:
    batch_size = 2
    multiplicity = 2
    encoder, inputs, r, coords_cond = _make_conditioning_case(batch_size, multiplicity)
    mask = torch.tensor([[True, False], [False, True]])
    q_uncond = _run_with_dynamics_condition(encoder, inputs, r, coords_cond, torch.zeros_like(mask))
    q_cond = _run_with_dynamics_condition(encoder, inputs, r, coords_cond, mask)

    selected = mask.reshape(-1)
    torch.testing.assert_close(q_cond[~selected], q_uncond[~selected])
    assert torch.all((q_cond[selected] - q_uncond[selected]).abs().sum(dim=(1, 2)) > 0)
