# SPDX-License-Identifier: Apache-2.0
"""MEM-short: short-horizon video memory for GR00T's SigLIP vision tower.

Implements the video-encoder design from "MEM: Multi-Scale Embodied Memory for VLA
Models" (arXiv:2603.03596), Section III-C / Appendix C: a ViT vision encoder is extended
to attend across K past+current frames by interleaving standard spatial (within-frame)
attention with a temporal (across-frame, fixed patch index, causal) attention pass every
`temporal_attn_every_n_layers` layers. The temporal pass reuses that SAME layer's
`SiglipAttention` Q/K/V/O projection weights -- no new learnable parameters are added
anywhere in this file. A fixed (non-learnable), pure-sine temporal position embedding is
added to patch embeddings once at the input, with the property that it is exactly zero at
offset 0 (the current frame) for every channel.

Frame ordering convention (enforced by construction, relied on throughout this file):
frames are ordered oldest -> newest, with the CURRENT frame always last (index T-1).
`video_frame_offsets` on the config must be an ascending list ending in 0, e.g.
`[-60, -40, -20, 0]`.

K=1 exact-reduction requirement (see plan): rather than relying on the temporal position
embedding's `e(0)=0` alone (which does NOT by itself make a single-key attention op an
identity -- `out_proj(v_proj(x))` is a nonzero learned affine map of `x`), both
`TemporalSiglipVisionEmbeddings.forward` and `TemporalSiglipEncoder.forward` explicitly
delegate to their stock parent-class implementation whenever the input has no separate
time axis (4D `(B,C,H,W)` pixel_values / 3D `(B,N,C)` embeddings). This means the K=1 case
runs the literal, unmodified stock code path -- not just a numerically-equivalent one.

Because `build_temporal_siglip_vision_model` returns a genuine `SiglipVisionModel`
instance (only its `.vision_model.{embeddings,encoder}` submodules are swapped, keeping
their parameter names/shapes identical to stock), `isinstance(..., SiglipVisionModel)`
checks elsewhere (e.g. LoRA target-module resolution in modeling_eagle2_5_vl.py) and
`state_dict()`/`load_state_dict(..., strict=True)` against a stock pretrained checkpoint
both work unmodified.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoder,
    SiglipVisionEmbeddings,
    SiglipVisionModel,
)


def sinusoidal_temporal_embedding(
    offsets: torch.Tensor, dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Fixed, non-learnable temporal position embedding, one row per frame offset.

    Pure-sine construction (`sin(offset * omega_j)` for every channel j) rather than the
    usual interleaved sin/cos scheme, because interleaved sin/cos has `cos(0) = 1`, which
    would violate the paper's `e(0) = 0` boundary condition. Pure sine is exactly zero at
    offset == 0 for every channel by construction.

    Args:
        offsets: 1D tensor of shape (T,), e.g. [-60, -40, -20, 0].
        dim: embedding dimension (matches the ViT's hidden_size).
    Returns:
        (T, dim) tensor, zero at whichever row has offset == 0.
    """
    offsets = offsets.to(dtype=torch.float32, device=device)
    freq_idx = torch.arange(dim, dtype=torch.float32, device=device)
    omega = 1.0 / (10000.0 ** (freq_idx / dim))
    angles = offsets[:, None] * omega[None, :]  # (T, dim)
    return torch.sin(angles).to(dtype)


def apply_temporal_attention(attn: SiglipAttention, x_bntc: torch.Tensor) -> torch.Tensor:
    """Causal multi-head attention over the time axis, reusing an existing layer's weights.

    Reuses `attn.{q,k,v,out}_proj` -- the SAME already-instantiated `nn.Linear` modules
    from the spatial `SiglipAttention` that just ran on this layer -- to compute Q/K/V/O
    for the temporal pass. No new parameters. Causal (offset-ascending order, so the
    current/last frame can attend to all past frames, past frames cannot see the future).

    Args:
        attn: the layer's existing SiglipAttention module.
        x_bntc: (B*N, T, C) -- one sequence per (batch, spatial-patch) pair.
    Returns:
        (B*N, T, C)
    """
    bn, t, c = x_bntc.shape
    q = attn.q_proj(x_bntc).view(bn, t, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x_bntc).view(bn, t, attn.num_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x_bntc).view(bn, t, attn.num_heads, attn.head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn.scale)
    out = out.transpose(1, 2).reshape(bn, t, c)
    return attn.out_proj(out)


class TemporalSiglipVisionEmbeddings(SiglipVisionEmbeddings):
    """Adds a fixed temporal position embedding on top of stock per-frame patch+spatial-pos embeddings.

    `__init__` is inherited unchanged from `SiglipVisionEmbeddings`, so `patch_embedding` /
    `position_embedding` / `position_ids` keep identical names and shapes to stock -- zero
    new parameters, full state_dict compatibility with a stock pretrained checkpoint.
    """

    def __init__(self, config):
        super().__init__(config)
        frame_offsets = list(getattr(config, "video_frame_offsets", [0.0]))
        assert frame_offsets[-1] == 0.0, (
            "video_frame_offsets must end in 0 (current frame last) -- "
            f"got {frame_offsets}"
        )
        self.register_buffer(
            "video_frame_offsets", torch.tensor(frame_offsets, dtype=torch.float32), persistent=False
        )

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        if pixel_values.dim() == 4:
            # Stock (B, C, H, W) single-frame path -- delegate to the literal parent
            # implementation, unmodified. This is what guarantees exact K=1 reduction.
            return super().forward(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        assert pixel_values.dim() == 5, f"expected (B,C,H,W) or (B,T,C,H,W), got {tuple(pixel_values.shape)}"
        b, t, c, h, w = pixel_values.shape
        assert t == self.video_frame_offsets.shape[0], (
            f"pixel_values has T={t} frames but config.video_frame_offsets has "
            f"{self.video_frame_offsets.shape[0]} entries ({self.video_frame_offsets.tolist()})"
        )
        flat = pixel_values.reshape(b * t, c, h, w)
        embeds = super().forward(flat, interpolate_pos_encoding=interpolate_pos_encoding)  # (B*T, N, D)
        n, d = embeds.shape[1], embeds.shape[2]
        embeds = embeds.view(b, t, n, d)
        if t > 1:
            temporal_pe = sinusoidal_temporal_embedding(
                self.video_frame_offsets, d, embeds.dtype, embeds.device
            )  # (T, D), zero at the current-frame row
            embeds = embeds + temporal_pe[None, :, None, :]
        return embeds  # (B, T, N, D)


class TemporalSiglipEncoder(SiglipEncoder):
    """Drop-in replacement for SiglipEncoder that interleaves temporal attention every Nth layer.

    `__init__` (beyond the new `temporal_attn_every_n_layers` int, stored as a plain python
    attribute -- not a buffer/parameter) is inherited unchanged, so `self.layers` holds the
    exact same 27 `SiglipEncoderLayer` instances with identical parameter names/shapes as
    stock `SiglipEncoder` -- zero new parameters.
    """

    def __init__(self, config, temporal_attn_every_n_layers: int = 4):
        super().__init__(config)
        self.temporal_attn_every_n_layers = temporal_attn_every_n_layers

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask=None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        if inputs_embeds.dim() == 3:
            # Stock (B, N, C) single-frame path -- delegate to the literal parent
            # implementation, unmodified (exact K=1 reduction guarantee).
            return super().forward(inputs_embeds, attention_mask, output_attentions, output_hidden_states)

        assert inputs_embeds.dim() == 4, f"expected (B,N,C) or (B,T,N,C), got {tuple(inputs_embeds.shape)}"
        assert attention_mask is None, "attention_mask is not supported in the multi-frame video-memory path"
        b, t, n, c = inputs_embeds.shape
        output_attentions = bool(
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = bool(
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds.reshape(b * t, n, c)
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states.view(b, t, n, c)[:, -1],)

            # -- spatial block: standard SiglipEncoderLayer block 1 (LN1 -> self_attn ->
            # residual), applied independently per (batch, frame) "image", unchanged. --
            residual = hidden_states
            x = layer.layer_norm1(hidden_states)
            attn_out, attn_weights = layer.self_attn(x, None, output_attentions=output_attentions)
            hidden_states = residual + attn_out
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

            # -- temporal block (every `temporal_attn_every_n_layers`th layer): additional
            # causal attention pass over the T axis at fixed patch index, reusing THIS
            # layer's self_attn weights, with its own residual add, inserted before the
            # MLP block -- see module docstring / plan Stage A / paper Appendix C.
            # Explicit `t > 1` guard (not numerically emergent): attention over a
            # length-1 sequence is NOT an identity -- out_proj(v_proj(x)) is a nonzero
            # learned affine map of x, even though softmax over one key is trivially 1.
            # Skipping this block entirely at t==1 is what makes that case bit-identical
            # to stock (verified in tests/test_temporal_siglip_k1_equivalence.py). --
            if t > 1 and layer_idx % self.temporal_attn_every_n_layers == 0:
                residual_t = hidden_states
                x_btnc = hidden_states.view(b, t, n, c)
                x_bn_t_c = x_btnc.permute(0, 2, 1, 3).reshape(b * n, t, c)
                temporal_out = apply_temporal_attention(layer.self_attn, x_bn_t_c)
                temporal_out = temporal_out.reshape(b, n, t, c).permute(0, 2, 1, 3).reshape(b * t, n, c)
                hidden_states = residual_t + temporal_out

            # -- MLP block: standard SiglipEncoderLayer block 2 (LN2 -> mlp -> residual),
            # unchanged. --
            residual = hidden_states
            x = layer.layer_norm2(hidden_states)
            x = layer.mlp(x)
            hidden_states = residual + x

        hidden_states = hidden_states.view(b, t, n, c)
        current = hidden_states[:, -1]  # (B, N, C) -- current-timestep tokens only
        if output_hidden_states:
            encoder_states = encoder_states + (current,)

        return BaseModelOutput(last_hidden_state=current, hidden_states=encoder_states, attentions=all_attentions)


def build_temporal_siglip_vision_model(config, temporal_attn_every_n_layers: int = 4) -> SiglipVisionModel:
    """Builds a SiglipVisionModel with temporal-attention-capable embeddings/encoder.

    `config.video_frame_offsets` (ascending, ending in 0) determines how many frames the
    returned model expects and what temporal position each carries; defaults to `[0.0]`
    (single current frame), which makes the returned model behave byte-identically to a
    stock `SiglipVisionModel` even before any pixel_values are passed in.
    """
    model = SiglipVisionModel(config)
    model.vision_model.embeddings = TemporalSiglipVisionEmbeddings(config)
    model.vision_model.encoder = TemporalSiglipEncoder(config, temporal_attn_every_n_layers=temporal_attn_every_n_layers)
    model.post_init()
    return model
