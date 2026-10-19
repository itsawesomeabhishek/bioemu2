import torch

from bioemu2.biomoa.ops.atom_token_ops import (
    broadcast_token_pair_to_atom_window,
    broadcast_tokens_to_atoms,
    gather_center_tokens_from_atomic,
    scatter_mean_atoms_to_tokens,
    window_atom_single_to_pair_keys,
    windowed_atom_pair_geometry,
)


def _ref_window_keys_loop(single, W, H):
    B, N, D = single.shape
    K = N // W
    ctx = (H - W) // 2
    out = torch.zeros(B, K, H, D, dtype=single.dtype, device=single.device)
    for b in range(B):
        for k in range(K):
            for j in range(H):
                g = k * W - ctx + j
                if 0 <= g < N:
                    out[b, k, j] = single[b, g]
    return out


def _random_ids(B, N, T, num_pad=0, device="cpu"):
    tok_idx = torch.randint(0, T, (B, N), device=device)
    atom_pad_mask = torch.ones(B, N, device=device)
    for batch_index in range(B):
        row_num_pad = max(num_pad - batch_index, 0)
        if row_num_pad > 0:
            atom_pad_mask[batch_index, -row_num_pad:] = 0.0
    return tok_idx, atom_pad_mask


def _broadcast_reference(per_token, tok_idx, atom_pad_mask):
    B, N = tok_idx.shape
    D = per_token.shape[-1]
    out = torch.zeros(B, N, D, dtype=per_token.dtype)
    for b in range(B):
        for n in range(N):
            if atom_pad_mask[b, n] != 0:
                out[b, n] = per_token[b, tok_idx[b, n]]
    return out


def _scatter_mean_reference(per_atom, tok_idx, atom_pad_mask, num_tokens):
    B, N = tok_idx.shape
    D = per_atom.shape[-1]
    out = torch.zeros(B, num_tokens, D, dtype=per_atom.dtype)
    for b in range(B):
        for t in range(num_tokens):
            total = torch.zeros(D, dtype=per_atom.dtype)
            count = 0.0
            for n in range(N):
                if atom_pad_mask[b, n] != 0 and int(tok_idx[b, n]) == t:
                    total = total + per_atom[b, n]
                    count += 1.0
            out[b, t] = total / (count + 1e-6)
    return out


def test_broadcast_matches_reference():
    B, N, T, D, num_pad = 3, 20, 6, 7, 3
    tok_idx, atom_pad_mask = _random_ids(B, N, T, num_pad=num_pad)
    per_token = torch.randn(B, T, D)
    got = broadcast_tokens_to_atoms(per_token, tok_idx, ~atom_pad_mask.bool())
    expected = _broadcast_reference(per_token, tok_idx, atom_pad_mask)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_broadcast_gradient_matches_reference():
    B, N, T, D = 2, 16, 5, 4
    tok_idx, atom_pad_mask = _random_ids(B, N, T, num_pad=3)
    x1 = torch.randn(B, T, D, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out_helper = broadcast_tokens_to_atoms(x1, tok_idx, ~atom_pad_mask.bool())
    out_ref = _broadcast_reference(x2, tok_idx, atom_pad_mask)
    g = torch.randn_like(out_helper)
    out_helper.backward(g)
    out_ref.backward(g)
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-6, rtol=1e-6)


def test_scatter_mean_matches_reference():
    B, N, T, D, num_pad = 3, 20, 6, 7, 3
    tok_idx, atom_pad_mask = _random_ids(B, N, T, num_pad=num_pad)
    per_atom = torch.randn(B, N, D)
    got = scatter_mean_atoms_to_tokens(per_atom, tok_idx, ~atom_pad_mask.bool(), num_tokens=T)
    expected = _scatter_mean_reference(per_atom, tok_idx, atom_pad_mask, num_tokens=T)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_scatter_mean_gradient_matches_reference():
    B, N, T, D = 2, 16, 5, 4
    tok_idx, atom_pad_mask = _random_ids(B, N, T, num_pad=3)
    x1 = torch.randn(B, N, D, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out_helper = scatter_mean_atoms_to_tokens(x1, tok_idx, ~atom_pad_mask.bool(), num_tokens=T)
    out_ref = _scatter_mean_reference(x2, tok_idx, atom_pad_mask, num_tokens=T)
    g = torch.randn_like(out_helper)
    out_helper.backward(g)
    out_ref.backward(g)
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)


def test_scatter_mean_empty_token_returns_zero():
    B, N, T, D = 1, 6, 4, 3
    tok_idx = torch.tensor([[0, 0, 1, 2, 2, 2]])
    atom_pad_mask = torch.ones(B, N)
    per_atom = torch.randn(B, N, D)
    got = scatter_mean_atoms_to_tokens(per_atom, tok_idx, ~atom_pad_mask.bool(), num_tokens=T)
    torch.testing.assert_close(got[:, 3], torch.zeros(B, D), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got[:, 0], per_atom[:, 0:2].mean(1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got[:, 1], per_atom[:, 2], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got[:, 2], per_atom[:, 3:6].mean(1), atol=1e-5, rtol=1e-5)


def test_negative_one_padding_matches_reference():
    B, N, T, D, num_pad = 2, 12, 4, 3, 5
    tok_idx, atom_pad_mask = _random_ids(B, N, T, num_pad=num_pad)
    tok_idx[~atom_pad_mask.bool()] = -1

    per_token = torch.randn(B, T, D)
    got_b = broadcast_tokens_to_atoms(per_token, tok_idx, ~atom_pad_mask.bool())
    exp_b = _broadcast_reference(per_token, tok_idx, atom_pad_mask)
    torch.testing.assert_close(got_b, exp_b, atol=1e-6, rtol=1e-6)

    per_atom = torch.randn(B, N, D)
    got_s = scatter_mean_atoms_to_tokens(per_atom, tok_idx, ~atom_pad_mask.bool(), num_tokens=T)
    exp_s = _scatter_mean_reference(per_atom, tok_idx, atom_pad_mask, num_tokens=T)
    torch.testing.assert_close(got_s, exp_s, atol=1e-5, rtol=1e-5)

    B2, T2, W, H, N2, D2, num_pad2 = 2, 16, 4, 8, 16, 8, 3
    z = torch.randn(B2, T2, T2, D2)
    tok_idx2, pad_mask2, tok_q, tok_k, pad_q, pad_k = _windowed_setup(B2, T2, W, H, N2, num_pad2)
    tok_idx2 = tok_idx2.clone()
    tok_idx2[~pad_mask2.bool()] = -1
    got_w = broadcast_token_pair_to_atom_window(
        z, atom_to_token_ids=tok_idx2, is_padded_atom=~pad_mask2.bool(), W=W, H=H
    )
    exp_w = _windowed_reference(z, tok_q, tok_k, pad_q, pad_k)
    torch.testing.assert_close(got_w, exp_w, atol=1e-5, rtol=1e-5)


def _random_atom_to_token(B, N, T, num_pad=0, device="cpu"):
    idx = torch.randint(0, T, (B, N), device=device)
    a2t = torch.nn.functional.one_hot(idx, T).float()
    if num_pad > 0:
        a2t[:, -num_pad:] = 0.0
    return a2t


def _windowed_setup(B, T, W, H, N, num_pad, device="cpu"):
    K = N // W
    a2t = _random_atom_to_token(B, N, T, num_pad=num_pad, device=device)
    tok_idx = a2t.argmax(-1)
    atom_pad_mask = a2t.sum(-1) != 0
    pad_mask = ~atom_pad_mask
    a2t_k = _ref_window_keys_loop(a2t.float(), W, H).view(B, K, H, T)
    tok_q, pad_q = tok_idx.view(B, K, W), pad_mask.view(B, K, W)
    tok_k, pad_k = a2t_k.argmax(-1), a2t_k.sum(-1) == 0
    return tok_idx, atom_pad_mask, tok_q, tok_k, pad_q, pad_k


def _windowed_reference(z, tok_q, tok_k, pad_q, pad_k):
    B, K, W = tok_q.shape
    H = tok_k.shape[-1]
    D = z.shape[-1]
    out = torch.zeros(B, K, W, H, D, dtype=z.dtype)
    for b in range(B):
        for w in range(K):
            for i in range(W):
                for j in range(H):
                    if not pad_q[b, w, i] and not pad_k[b, w, j]:
                        out[b, w, i, j] = z[b, tok_q[b, w, i], tok_k[b, w, j]]
    return out


def test_windowed_pair_matches_reference():
    B, T, W, H, N, D, num_pad = 2, 16, 4, 8, 16, 32, 3
    z = torch.randn(B, T, T, D)
    tok_idx, atom_pad_mask, tok_q, tok_k, pad_q, pad_k = _windowed_setup(B, T, W, H, N, num_pad)
    got = broadcast_token_pair_to_atom_window(
        z, atom_to_token_ids=tok_idx, is_padded_atom=~atom_pad_mask.bool(), W=W, H=H
    )
    expected = _windowed_reference(z, tok_q, tok_k, pad_q, pad_k)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_windowed_pair_gradient_matches_reference():
    B, T, W, H, N, D, num_pad = 2, 8, 4, 8, 8, 16, 3
    tok_idx, atom_pad_mask, tok_q, tok_k, pad_q, pad_k = _windowed_setup(B, T, W, H, N, num_pad)
    z1 = torch.randn(B, T, T, D, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    out_helper = broadcast_token_pair_to_atom_window(
        z1, atom_to_token_ids=tok_idx, is_padded_atom=~atom_pad_mask.bool(), W=W, H=H
    )
    out_ref = _windowed_reference(z2, tok_q, tok_k, pad_q, pad_k)
    g = torch.randn_like(out_helper)
    out_helper.backward(g)
    out_ref.backward(g)
    torch.testing.assert_close(z1.grad, z2.grad, atol=1e-5, rtol=1e-5)


def test_broadcast_multiplicity_matches_naive_repeat():
    B, N, T, D, M = 2, 10, 4, 3, 3
    tok_idx, atom_pad_mask = _random_ids(B, N, T, num_pad=2)
    per_token = torch.randn(B * M, T, D)
    got = broadcast_tokens_to_atoms(per_token, tok_idx, ~atom_pad_mask.bool(), multiplicity=M)
    expected = _broadcast_reference(
        per_token, tok_idx.repeat_interleave(M, 0), atom_pad_mask.repeat_interleave(M, 0)
    )
    torch.testing.assert_close(got, expected)


def test_scatter_mean_multiplicity_matches_naive_repeat():
    B, N, T, D, M = 2, 10, 4, 3, 3
    tok_idx, atom_pad_mask = _random_ids(B, N, T, num_pad=2)
    per_atom = torch.randn(B * M, N, D)
    got = scatter_mean_atoms_to_tokens(
        per_atom, tok_idx, ~atom_pad_mask.bool(), num_tokens=T, multiplicity=M
    )
    expected = _scatter_mean_reference(
        per_atom,
        tok_idx.repeat_interleave(M, 0),
        atom_pad_mask.repeat_interleave(M, 0),
        num_tokens=T,
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_scatter_mean_excludes_inter_sample_padding():
    B, N, T, D = 1, 6, 3, 2
    tok_idx = torch.tensor([[0, 0, 1, 2, 0, 0]])
    atom_pad_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    per_atom = torch.arange(1.0, N + 1).view(1, N, 1).expand(B, N, D).contiguous()
    out = scatter_mean_atoms_to_tokens(per_atom, tok_idx, ~atom_pad_mask.bool(), num_tokens=T)
    torch.testing.assert_close(out[0, 0], torch.full((D,), 1.5))


def _windowed_pair_reference(atom_ref_pos, atom_mask, atom_uid, W, H):
    B, N, _ = atom_ref_pos.shape
    K = N // W
    offset = (H - W) // 2
    d = torch.zeros(B, K, W, H, 3, dtype=atom_ref_pos.dtype)
    v = torch.zeros(B, K, W, H, 1, dtype=atom_ref_pos.dtype)
    for b in range(B):
        for k in range(K):
            for i in range(W):
                q = k * W + i
                for j in range(H):
                    g = k * W - offset + j
                    valid_key = 0 <= g < N
                    key_pos = atom_ref_pos[b, g] if valid_key else torch.zeros(3)
                    d[b, k, i, j] = key_pos - atom_ref_pos[b, q]
                    mask_k = bool(atom_mask[b, g]) if valid_key else False
                    uid_k = int(atom_uid[b, g]) if valid_key else 0
                    same = bool(atom_mask[b, q]) and mask_k and int(atom_uid[b, q]) == uid_k
                    v[b, k, i, j, 0] = 1.0 if same else 0.0
    d_norm = 1.0 / (1.0 + (d * d).sum(-1, keepdim=True))
    return d, d_norm, v


def test_windowed_atom_pair_geometry_matches_reference():
    B, N, W, H, num_pad = 2, 16, 4, 8, 3
    torch.manual_seed(0)
    atom_ref_pos = torch.randn(B, N, 3)
    atom_mask = torch.ones(B, N)
    if num_pad > 0:
        atom_mask[:, -num_pad:] = 0.0
    atom_uid = torch.randint(0, 3, (B, N)).float()

    d, d_norm, v = windowed_atom_pair_geometry(
        atom_ref_pos, atom_mask, atom_uid, W, H, out_dtype=torch.float32
    )
    d_ref, d_norm_ref, v_ref = _windowed_pair_reference(atom_ref_pos, atom_mask, atom_uid, W, H)

    torch.testing.assert_close(v, v_ref, atol=0, rtol=0)
    torch.testing.assert_close(d * v, d_ref * v_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(d_norm * v, d_norm_ref * v_ref, atol=1e-6, rtol=1e-6)


def test_windowed_atom_pair_geometry_gradient_matches_reference():
    torch.manual_seed(0)
    B, N, W, H = 2, 16, 4, 8
    atom_mask = torch.ones(B, N)
    atom_mask[:, -3:] = 0.0
    atom_uid = torch.randint(0, 3, (B, N)).float()
    x1 = torch.randn(B, N, 3, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    d1, dn1, v1 = windowed_atom_pair_geometry(
        x1, atom_mask, atom_uid, W, H, out_dtype=torch.float32
    )
    d2, dn2, v2 = _windowed_pair_reference(x2, atom_mask, atom_uid, W, H)
    # Masked geometry slots carry no physical signal.
    g_d, g_dn = torch.randn_like(d1), torch.randn_like(dn1)
    (((d1 * v1) * g_d).sum() + ((dn1 * v1) * g_dn).sum()).backward()
    (((d2 * v2) * g_d).sum() + ((dn2 * v2) * g_dn).sum()).backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)


def test_window_atom_single_to_pair_keys_matches_loop_reference():
    B, N, W, H, D = 2, 16, 4, 8, 3
    torch.manual_seed(0)
    atom_single = torch.randn(B, N, D, dtype=torch.float64)
    ref = _ref_window_keys_loop(atom_single, W, H)
    got = window_atom_single_to_pair_keys(atom_single, W, H)
    torch.testing.assert_close(got, ref.unsqueeze(2), atol=0, rtol=0)


def test_window_atom_single_to_pair_keys_gradient_matches_loop_reference():
    torch.manual_seed(0)
    B, N, W, H, D = 2, 16, 4, 8, 5
    x1 = torch.randn(B, N, D, dtype=torch.float64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out1 = window_atom_single_to_pair_keys(x1, W, H)
    out2 = _ref_window_keys_loop(x2, W, H)
    g = torch.randn_like(out2)
    out1.backward(g.unsqueeze(2))
    out2.backward(g)
    torch.testing.assert_close(x1.grad, x2.grad, atol=0, rtol=0)


class TestGatherCenterTokensFromAtomic:
    def test_multiplicity_and_padding_mapping(self) -> None:
        multiplicity = 2
        B, N_atoms, D, N_token = 2, 6, 4, 3
        n_rows = B * multiplicity
        rows = torch.arange(n_rows).view(n_rows, 1, 1)
        atoms = torch.arange(N_atoms).view(1, N_atoms, 1)
        x_atomic = (rows * 100 + atoms).float().expand(n_rows, N_atoms, D).contiguous()
        token_center_atom_index = torch.tensor([[0, 2, 4], [1, 3, 5]])
        token_mask = torch.tensor([[True, True, False], [True, False, True]])

        out = gather_center_tokens_from_atomic(
            x_atomic, token_center_atom_index, token_mask, multiplicity=multiplicity
        )

        row_base = torch.arange(n_rows).view(n_rows, 1) * 100
        idx_per_row = token_center_atom_index.repeat_interleave(multiplicity, 0)
        mask_per_row = token_mask.repeat_interleave(multiplicity, 0).unsqueeze(-1)
        expected = ((row_base + idx_per_row).unsqueeze(-1) * mask_per_row).float()
        expected = expected.expand(n_rows, N_token, D)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)
