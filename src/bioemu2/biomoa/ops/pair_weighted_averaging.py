# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

import torch


class PairWeightedAveraging(torch.nn.Module):
    """Pair-weighted averaging layer for updating MSA representations from pair features."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e6,
        use_bias_in_layer_norm: bool = True,
    ) -> None:
        """Initialize the pair weighted averaging layer.

        Parameters
        ----------
        c_m: int
            The dimension of the input sequence.
        c_z: int
            The dimension of the input pairwise tensor.
        c_h: int
            The dimension of the hidden.
        num_heads: int
            The number of heads.
        inf: float
            The value to use for masking, default 1e6.
        use_bias_in_layer_norm: bool
            Whether to use bias in the pair LayerNorm. When False, the LN gamma
            can be fused into a conv2d weight for efficiency.

        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf
        self.use_bias_in_layer_norm = use_bias_in_layer_norm

        self.norm_m = torch.nn.LayerNorm(c_m)
        self.norm_z = torch.nn.LayerNorm(c_z, bias=use_bias_in_layer_norm)

        self.proj_m = torch.nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = torch.nn.Linear(c_m, c_h * num_heads, bias=False)
        # Pair bias projection: separate LN + Linear so we can fuse them via
        # F.conv2d in the non-chunked path, while still
        # accessing the linear weight per-head in the chunked path.
        self.proj_z = torch.nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = torch.nn.Linear(c_h * num_heads, c_m, bias=False)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_heads: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.torch.Tensor
            The input sequence tensor (B, S, N, D)
        z : torch.torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.torch.Tensor
            The output sequence tensor (B, S, N, D)

        """
        # Compute layer norms
        m = self.norm_m(m)

        if chunk_heads and not self.training:
            z = self.norm_z(z)  # full LayerNorm (normalize + gamma)
            # Compute heads sequentially
            o_chunks = []
            for head_idx in range(self.num_heads):
                sliced_weight_proj_m = self.proj_m.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_g = self.proj_g.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_z = self.proj_z.weight[head_idx : (head_idx + 1), :]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, head_idx * self.c_h : (head_idx + 1) * self.c_h
                ]

                # Project input tensors
                v: torch.Tensor = m @ sliced_weight_proj_m.T
                v = v.reshape(*v.shape[:3], 1, self.c_h)
                v = v.permute(0, 3, 1, 2, 4)

                # Compute weights
                b: torch.Tensor = z @ sliced_weight_proj_z.T
                b = b.permute(0, 3, 1, 2).float()
                b.masked_fill_(~mask[:, None].bool(), -self.inf)
                w = torch.softmax(b, dim=-1)

                # Compute gating
                g: torch.Tensor = m @ sliced_weight_proj_g.T
                g = g.sigmoid()

                # Compute output
                o = torch.einsum("bhij,bhsjd->bhsid", w, v)
                o = o.permute(0, 2, 3, 1, 4)
                o = o.reshape(*o.shape[:3], 1 * self.c_h)
                o_chunks = g * o
                if head_idx == 0:
                    o_out = o_chunks @ sliced_weight_proj_o.T
                else:
                    o_out += o_chunks @ sliced_weight_proj_o.T
            return o_out
        else:
            # Project input tensors
            v = self.proj_m(m)
            v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            if self.use_bias_in_layer_norm:
                z = self.norm_z(z)  # [B, N, N, c_z]
                b = self.proj_z(z)  # [B, N, N, H]
                b = b.permute(0, 3, 1, 2).float()  # [B, H, N, N]
            else:
                # Fused LN + Linear + permute via 1×1 conv2d.
                # Only safe when LN has no bias — gamma is folded into the conv weight.
                z_normed = torch.nn.functional.layer_norm(z, (self.c_z,))
                z_cf = z_normed.permute(0, 3, 1, 2)  # [B, c_z, N, N]
                weight = (self.proj_z.weight * self.norm_z.weight[None, :])[:, :, None, None]
                b = torch.nn.functional.conv2d(z_cf, weight).float()  # [B, H, N, N]
            b.masked_fill_(~mask[:, None].bool(), -self.inf)
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g = self.proj_g(m)
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
            o = self.proj_o(g * o)
            return o
