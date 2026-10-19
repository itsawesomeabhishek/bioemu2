"""Canonical BioMoa encoder tests."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.modules.encoders import AtomAttentionEncoder
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.modules.encoders import (
    DistributedAtomAttentionEncoder,
)
from bioemu2.context_parallel.utils.dtensor import (
    PL_SEQ,
    gather_along_atom,
    gather_along_window,
    shard_along_atom,
    shard_pair_along_token,
)
from bioemu2.context_parallel.utils.padding import (
    pad_along_atom,
    pad_along_token,
    pad_amounts,
    pad_pair_along_token,
    unpad_along_atom,
    unpad_along_window,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
)

BATCH_SIZE = 1
N_TOKENS = 7
N_ATOMS = 96
WINDOW = 32


def _encoder_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
) -> None:
    torch.manual_seed(1234)
    serial = (
        AtomAttentionEncoder(
            atom_s=32,
            atom_z=16,
            token_s=32,
            token_z=32,
            atoms_per_window_queries=WINDOW,
            atoms_per_window_keys=WINDOW,
            structure_prediction=True,
            use_atom_encoder=False,
            use_atom_to_token_aggregation=False,
        )
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    generator = torch.Generator(device=device).manual_seed(5678)
    ref_pos = torch.randn(BATCH_SIZE, N_ATOMS, 3, generator=generator, device=device)
    ref_charge = torch.randn(BATCH_SIZE, N_ATOMS, generator=generator, device=device)
    ref_element = torch.randint(0, 128, (BATCH_SIZE, N_ATOMS), generator=generator, device=device)
    ref_atom_name_chars = torch.randint(
        0, 64, (BATCH_SIZE, N_ATOMS, 4), generator=generator, device=device
    )
    ref_space_uid = torch.arange(N_ATOMS, device=device).unsqueeze(0) // 4
    atom_pad_mask = torch.ones(BATCH_SIZE, N_ATOMS, dtype=torch.bool, device=device)
    atom_pad_mask[:, -5:] = False
    atom_to_token_ids = (
        torch.arange(N_ATOMS, device=device).mul(N_TOKENS).div(N_ATOMS, rounding_mode="floor")
    ).unsqueeze(0)
    atom_to_token_ids[:, -5:] = 0
    token_pad_mask = torch.ones(BATCH_SIZE, N_TOKENS, dtype=torch.bool, device=device)
    s_trunk = torch.randn(BATCH_SIZE, N_TOKENS, 32, generator=generator, device=device)
    z = torch.randn(BATCH_SIZE, N_TOKENS, N_TOKENS, 32, generator=generator, device=device)
    r = torch.randn(BATCH_SIZE * 2, N_ATOMS, 3, generator=generator, device=device)
    for tensor in [
        ref_pos,
        ref_charge,
        ref_element,
        ref_atom_name_chars,
        ref_space_uid,
        atom_pad_mask,
        atom_to_token_ids,
        token_pad_mask,
        s_trunk,
        r,
    ]:
        dist.broadcast(tensor, src=0)

    with torch.no_grad():
        _, q_expected, c_expected, p_expected = serial(
            atom_ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_ref_space_uid=ref_space_uid,
            atom_pad_mask=atom_pad_mask,
            atom_to_token_ids=atom_to_token_ids,
            token_pad_mask=token_pad_mask,
            s_trunk=s_trunk,
            z=z,
            r=r,
            multiplicity=2,
        )

    sizes = pad_amounts(N_TOKENS, N_ATOMS, n_msa=0, W=WINDOW, mesh=mesh)
    atom_inputs = pad_along_atom(
        [
            ref_pos,
            ref_charge,
            ref_element,
            ref_atom_name_chars,
            ref_space_uid,
            atom_pad_mask,
            atom_to_token_ids,
            r,
        ],
        sizes["n_atom_pad"],
    )
    (
        ref_pos_p,
        ref_charge_p,
        ref_element_p,
        ref_atom_name_chars_p,
        ref_space_uid_p,
        atom_pad_mask_p,
        atom_to_token_ids_p,
        r_p,
    ) = atom_inputs
    s_trunk_p = pad_along_token([s_trunk], sizes["n_token_pad"])[0]
    z_p = pad_pair_along_token([z], sizes["n_token_pad"])[0]
    atom_to_token_ids_dt, atom_valid_mask_dt = shard_along_atom(
        [atom_to_token_ids_p, atom_pad_mask_p], mesh, src_data_rank=None
    )
    (z_dt,) = shard_pair_along_token([z_p], mesh, src_data_rank=None)

    distributed = DistributedAtomAttentionEncoder(serial, mesh).to(device).eval()
    with torch.no_grad():
        static = distributed.compute_static_context(
            atom_ref_pos=ref_pos_p,
            ref_charge=ref_charge_p,
            ref_element=ref_element_p,
            ref_atom_name_chars=ref_atom_name_chars_p,
            atom_ref_space_uid=ref_space_uid_p,
            atom_pad_mask=atom_pad_mask_p,
            atom_to_token_ids=atom_to_token_ids_p,
            atom_to_token_ids_dt=atom_to_token_ids_dt,
            atom_valid_mask_dt=atom_valid_mask_dt,
            s_trunk=s_trunk_p,
            z=z_dt,
        )
        q_dt, c_dt, p_dt = distributed.apply_dynamic(static, r=r_p, multiplicity=2)

    for label, output in (("q", q_dt), ("c", c_dt), ("p", p_dt)):
        assert_output_placement(output, expected_placements=PL_SEQ, label=f"BioMoaEncoder/{label}")
        assert_replicate_values_against_rank0(output)
    q = unpad_along_atom(gather_along_atom([q_dt]), N_ATOMS)[0]
    c = unpad_along_atom(gather_along_atom([c_dt]), N_ATOMS)[0]
    p = unpad_along_window(gather_along_window([p_dt]), n_real=N_ATOMS // WINDOW)[0]
    gather_n_assert_parity_at_rank0(q_expected, q, label="BioMoaEncoder/q", atol=5e-5)
    gather_n_assert_parity_at_rank0(c_expected, c, label="BioMoaEncoder/c", atol=5e-5)
    gather_n_assert_parity_at_rank0(p_expected, p, label="BioMoaEncoder/p", atol=5e-5)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
def test_encoder_parity() -> None:
    spawn_cp_test(_encoder_parity)
