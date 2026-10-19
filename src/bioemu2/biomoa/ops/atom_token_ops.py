"""Index-based token/atom gather-scatter and sliding-window atom-pair operations."""

import torch


def broadcast_tokens_to_atoms(
    per_token: torch.Tensor,  # (B*M, T, D) per-token features
    atom_to_token_ids: torch.Tensor,  # (B, N) token id per atom
    is_padded_atom: torch.Tensor,  # (B, N) bool, True on padding atoms
    multiplicity: int = 1,  # M: diffusion samples folded into the batch
) -> torch.Tensor:
    """Broadcast per-token features onto atoms via index gather, zeroed on padding atoms.

    Pad atoms are routed to an appended zero token row (index ``T``) so the gather itself zeroes
    them, avoiding a second full-size masked_fill over the ``(B, M, N, D)`` output. The shared
    ``(B, N)`` index and pad mask are broadcast over the ``M`` replica axis.
    """
    B, N = atom_to_token_ids.shape
    M = multiplicity
    T, D = per_token.shape[-2], per_token.shape[-1]
    per_token = torch.nn.functional.pad(
        per_token.view(B, M, T, D), (0, 0, 0, 1)
    )  # (B, M, T + 1, D), row T is 0
    routed_ids = atom_to_token_ids.masked_fill(is_padded_atom, T)  # pads -> zero row
    idx = routed_ids.view(B, 1, N, 1).expand(B, M, N, D)
    return torch.gather(per_token, 2, idx).reshape(B * M, N, D)  # (B * M, N, D)


def scatter_mean_atoms_to_tokens(
    per_atom: torch.Tensor,  # (B*M, N, D) per-atom features
    atom_to_token_ids: torch.Tensor,  # (B, N) token id per atom
    is_padded_atom: torch.Tensor,  # (B, N) bool, True on padding atoms
    num_tokens: int,  # T: number of output token slots
    multiplicity: int = 1,  # M: diffusion samples folded into the batch
) -> torch.Tensor:
    """Aggregate per-atom features into per-token means via scatter-add, excluding padding atoms.

    Pad atoms are routed to a dump token slot (index ``num_tokens``) that is sliced off, so both the
    summed values and the per-token counts exclude them. The scatter-add runs in true float32
    (autocast disabled) for numerical stability; the shared ``(B, N)`` index and pad mask are
    broadcast over the ``M`` replica axis.
    """
    B, N = atom_to_token_ids.shape
    M = multiplicity
    D = per_atom.shape[-1]
    routed_ids = atom_to_token_ids.masked_fill(is_padded_atom, num_tokens)  # pads -> dump slot
    idx = routed_ids.view(B, 1, N, 1).expand(B, M, N, D)
    agg = torch.zeros(B, M, num_tokens + 1, D, dtype=torch.float32, device=per_atom.device)
    with torch.autocast("cuda", enabled=False):
        agg.scatter_add_(2, idx, per_atom.view(B, M, N, D).float())  # (B, M, num_tokens + 1, D)
    counts = torch.zeros(B, num_tokens + 1, dtype=per_atom.dtype, device=per_atom.device)
    counts.scatter_add_(1, routed_ids, torch.ones_like(routed_ids, dtype=per_atom.dtype))
    agg, counts = agg[:, :, :num_tokens], counts[:, :num_tokens]  # drop the dump slot
    return (agg / (counts.view(B, 1, num_tokens, 1) + 1e-6)).reshape(B * M, num_tokens, D)


def gather_center_tokens_from_atomic(
    x_atomic: torch.Tensor,  # (B*M, N, D) per-atom features
    token_center_atom_index: torch.Tensor,  # (B, T) center-atom index per token
    token_mask: torch.Tensor,  # (B, T) bool, True on real tokens (False on padding)
    multiplicity: int = 1,  # M: diffusion samples folded into the batch
) -> torch.Tensor:  # (B*M, T, D)
    """Gather each token's center-atom feature, zeroed on padding tokens.

    ``out[t] = x_atomic[token_center_atom_index[t]]`` over the ``T`` tokens, the complement of
    ``broadcast_tokens_to_atoms``. The shared ``(B, T)`` index and mask are broadcast over the ``M``
    replica axis rather than materialised to ``(B*M, T)``.
    """
    B, T = token_center_atom_index.shape
    M = multiplicity
    N, D = x_atomic.shape[-2], x_atomic.shape[-1]
    idx = token_center_atom_index.view(B, 1, T, 1).expand(B, M, T, D)
    x_token = torch.gather(x_atomic.view(B, M, N, D), 2, idx)  # (B, M, T, D)
    x_token = x_token.masked_fill(~token_mask.view(B, 1, T, 1), 0.0)  # zero padded tokens
    return x_token.reshape(B * M, T, D)


def broadcast_token_pair_to_atom_window(
    z: torch.Tensor,  # (B, T, T, D) token-pair features
    atom_to_token_ids: torch.Tensor,  # (B, N) token id per atom
    is_padded_atom: torch.Tensor,  # (B, N) bool, True on padding atoms
    W: int,  # query window width, must divide N
    H: int,  # key window width, H - W must be even and non-negative
) -> torch.Tensor:
    """Broadcast token-pair features ``z`` onto sliding windows of atom pairs, zeroed on padding.

    ``out[k, w, l] = z[tok_q[k, w], tok_k[k, l]]`` over ``NW = N // W`` query windows of ``W`` atoms,
    each attending an ``H``-atom key neighbourhood (key side via ``pad`` + ``unfold``, step ``W``,
    size ``H``); padded query/key positions are zeroed. Index form of ``"bijd,bwki,bwlj->bwkld"``.

    The two token axes of ``z`` are flattened into one and gathered with a single combined index
    rather than 3-axis advanced indexing ``z[b, tok_q, tok_k]``. Gather's backward is a 2-D
    scatter_add into ``(B, T*T, D)`` whereas advanced-indexing's is a 3-D index_put accumulate, so
    this is ~35% faster forward+backward at ``atom_z=16``.
    """
    B, N = atom_to_token_ids.shape
    assert z.shape[1] == z.shape[2], f"z token axes must be square, got {z.shape[1:3]}."
    assert N % W == 0, f"Number of atoms {N} is not divisible by window size {W}."
    assert (H - W) >= 0 and (H - W) % 2 == 0, f"H - W = {H - W} must be even and non-negative."
    assert is_padded_atom.shape == atom_to_token_ids.shape
    NW = N // W
    context = (H - W) // 2  # key context atoms on each side of a query window

    # Window the flat atom->token index / pad mask into query windows (B, NW, W).
    tok_q = atom_to_token_ids.view(B, NW, W)
    pad_q = is_padded_atom.view(B, NW, W)

    # Key side: pad by `context` (index -1 / pad True on out-of-window slots), unfold to (B, NW, H).
    tok_k = torch.nn.functional.pad(atom_to_token_ids, (context, context), value=-1).unfold(
        dimension=1, size=H, step=W
    )
    pad_k = torch.nn.functional.pad(is_padded_atom, (context, context), value=True).unfold(
        dimension=1, size=H, step=W
    )

    # Combine the two clamped token indices into one flat index (pad/out-of-window slots carry -1,
    # so clamp them to a valid token; their output is zeroed by pad_q / pad_k below).
    D = z.shape[-1]
    z_flat = z.reshape(B, z.shape[1] * z.shape[2], D)  # (B, T*T, D), contiguous view (no copy)
    flat_idx = tok_q.clamp_min(0).unsqueeze(3) * z.shape[1] + tok_k.clamp_min(0).unsqueeze(2)
    flat_idx = flat_idx.reshape(B, NW * W * H, 1).expand(B, NW * W * H, D)  # (B, NW*W*H, D)
    # gather returns a fresh copy (never a view of z), so zeroing pads in place is safe and avoids a
    # second full (B, NW, W, H, D) allocation. Query/key masks are applied separately so the combined
    # (B, NW, W, H) bool is never materialized.
    out = torch.gather(z_flat, 1, flat_idx).view(B, NW, W, H, D)  # (B, NW, W, H, D)
    out.masked_fill_(pad_q.view(B, NW, W, 1, 1), 0.0)
    out.masked_fill_(pad_k.view(B, NW, 1, H, 1), 0.0)
    return out


def window_atom_single_to_pair_keys(
    atom_single: torch.Tensor,  # (B, N, D) per-atom single features
    atoms_per_window_queries: int,  # W: query window width, must divide N
    atoms_per_window_keys: int,  # H: key window width, H - W must be even and non-negative
) -> torch.Tensor:  # (B, NW, 1, H, D)
    """Key side of the windowed atom-pair tensor from the atom single repr.

    ``keys[k, j] = atom_single[k*W - context + j]`` with ``context = (H - W) // 2``, via zero-``pad``
    + ``unfold`` (size ``H``, step ``W``), plus a singleton query axis so it broadcasts over the
    ``W`` queries. Out-of-range slots come from zero padding.
    """
    B, N, _ = atom_single.shape
    W, H = atoms_per_window_queries, atoms_per_window_keys
    assert N % W == 0, f"Number of atoms {N} is not divisible by window size {W}."
    context = (H - W) // 2  # key context atoms on each side of a query window
    # keys[k, j] = atom_single[k*W - context + j]; boundary slots land on the zero pad.
    padded = torch.nn.functional.pad(atom_single, (0, 0, context, context))  # (B, N + 2*ctx, D)
    keys = padded.unfold(dimension=1, size=H, step=W)  # (B, NW, D, H)
    return keys.transpose(-1, -2).unsqueeze(2)  # (B, NW, 1, H, D), query axis broadcasts over W


def windowed_atom_pair_geometry(
    atom_ref_pos: torch.Tensor,  # (B, N, 3) reference atom coordinates (angstrom)
    atom_mask: torch.Tensor,  # (B, N) 1.0 on real atoms, 0.0 on padding
    atom_uid: torch.Tensor,  # (B, N) integer reference-space uid per atom
    atoms_per_window_queries: int,  # W: query window width, must be even and divide N
    atoms_per_window_keys: int,  # H: key window width
    out_dtype: torch.dtype,  # dtype of the returned pair mask (the atom single-repr dtype)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sliding-window atom-atom pair geometry for the atom-attention encoder.

    Atoms split into ``NW = N // W`` query windows of width ``W``, each attending an ``H``-atom key
    window (same ``pad`` + ``unfold`` key gather as ``window_atom_single_to_pair_keys``, applied to
    positions, mask, and uid).

    Returns:
        atom_pair_disp: key-minus-query displacements, ``(B, NW, W, H, 3)``.
        atom_pair_inv_dist: inverse-square weight ``1 / (1 + ||disp||^2)``, ``(B, NW, W, H, 1)``.
        atom_pair_is_same_ref_space: whether both atoms are real and share the same
            reference-space uid, ``(B, NW, W, H, 1)``.
    """
    B, N, _ = atom_ref_pos.shape
    W, H = atoms_per_window_queries, atoms_per_window_keys
    assert N % W == 0, f"Number of atoms {N} is not divisible by window size {W}."
    NW = N // W  # number of query windows
    assert W % 2 == 0, f"Query window {W} must be even."
    assert H % W == 0, f"Key window {H} must be a multiple of the query window {W}."

    # keys[k, j] = atom[k*W - context + j] via pad+unfold; out-of-window slots hit the pad.
    context = (H - W) // 2
    pad = torch.nn.functional.pad

    atom_ref_pos_keys = (
        pad(atom_ref_pos, (0, 0, context, context))  # (B, N + 2*ctx, 3)
        .unfold(dimension=1, size=H, step=W)  # (B, NW, 3, H)
        .transpose(-1, -2)  # (B, NW, H, 3)
        .unsqueeze(2)  # (B, NW, 1, H, 3), query axis broadcasts over W
    )
    atom_ref_pos_queries = atom_ref_pos.view(B, NW, W, 1, 3)  # (B, NW, W, 1, 3)

    atom_pair_disp = atom_ref_pos_keys - atom_ref_pos_queries  # (B, NW, W, H, 3)
    sq_dist = torch.sum(atom_pair_disp * atom_pair_disp, dim=-1, keepdim=True)  # (B, NW, W, H, 1)
    atom_pair_inv_dist = 1 / (1 + sq_dist)  # (B, NW, W, H, 1)

    # Fold key mask into key uid: pad/out-of-window slots get UID_PAD=-1 (never a real uid>=0), so
    # they drop from the mask. Query mask stays separate — a both-side sentinel would self-match
    # (-1==-1).
    atom_mask_queries = atom_mask.view(B, NW, W, 1).bool()  # (B, NW, W, 1)
    atom_uid_queries = atom_uid.view(B, NW, W, 1).long()  # (B, NW, W, 1)

    UID_PAD = -1
    masked_uid = torch.where(atom_mask.bool(), atom_uid.long(), UID_PAD)  # (B, N)
    masked_uid_keys = (
        pad(masked_uid, (context, context), value=UID_PAD)
        .unfold(dimension=1, size=H, step=W)
        .view(B, NW, 1, H)
    )  # (B, NW, 1, H)

    atom_pair_is_same_ref_space = (
        (atom_mask_queries & (atom_uid_queries == masked_uid_keys)).unsqueeze(-1).to(out_dtype)
    )  # (B, NW, W, H, 1)
    return atom_pair_disp, atom_pair_inv_dist, atom_pair_is_same_ref_space
