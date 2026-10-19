# SPDX-License-Identifier: Apache-2.0
"""MEM-short: short-horizon video memory for GR00T's SigLIP vision tower.

Implements the video-encoder design from "MEM: Multi-Scale Embodied Memory for VLA
Models" (arXiv:2603.03596), Section III-C / Appendix C: a ViT vision encoder is extended
to attend across K past+current frames with space-time separable attention. Every
`temporal_attn_every_n_layers`th layer first adds a fixed sinusoidal temporal position
embedding e(t) (e(0)=0) to its input, then runs a causal temporal attention pass (across
frames, fixed patch index) followed by the standard spatial (within-frame) attention, both
reusing that SAME layer's LayerNorm and `SiglipAttention` Q/K/V/O weights -- no new
learnable parameters are added anywhere in this file. Past-frame tokens are dropped after
the last temporal layer, so only current-frame tokens reach the VLM backbone.

Frame ordering convention (enforced by construction, relied on throughout this file):
frames are ordered oldest -> newest, with the CURRENT frame always last (index T-1).
`video_frame_offsets` on the config must be an ascending list ending in 0, e.g.
`[-60, -40, -20, 0]`.

K=1 exact-reduction requirement: rather than relying on the temporal position
embedding's `e(0)=0` alone (which does NOT by itself make a single-key attention op an
identity -- `out_proj(v_proj(x))` is a nonzero learned affine map of `x`), both
`TemporalSiglipVisionEmbeddings.forward` and `TemporalSiglipEncoder.forward` explicitly
delegate to their stock parent-class implementation whenever the input has no separate
time axis (4D `(B,C,H,W)` pixel_values / 3D `(B,N,C)` embeddings) or T == 1. This means the K=1 case
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
    """Per-frame stock patch+spatial-pos embeddings for a (B, T, C, H, W) frame stack.

    `__init__` is inherited unchanged from `SiglipVisionEmbeddings`, so `patch_embedding` /
    `position_embedding` / `position_ids` keep identical names and shapes to stock -- zero
    new parameters, full state_dict compatibility with a stock pretrained checkpoint.

    The temporal position embedding e(t) is NOT added here: per the paper's Appendix C it is
    added to the input of every temporal-attention layer (see TemporalSiglipEncoder).
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_frames = len(getattr(config, "video_frame_offsets", [0.0]))

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        if pixel_values.dim() == 4:
            # Stock (B, C, H, W) single-frame path -- delegate to the literal parent
            # implementation, unmodified. This is what guarantees exact K=1 reduction.
            return super().forward(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        assert pixel_values.dim() == 5, f"expected (B,C,H,W) or (B,T,C,H,W), got {tuple(pixel_values.shape)}"
        b, t, c, h, w = pixel_values.shape
        assert t == self.num_frames, (
            f"pixel_values has T={t} frames but config.video_frame_offsets has {self.num_frames} entries"
        )
        flat = pixel_values.reshape(b * t, c, h, w)
        embeds = super().forward(flat, interpolate_pos_encoding=interpolate_pos_encoding)  # (B*T, N, D)
        return embeds.view(b, t, embeds.shape[1], embeds.shape[2])  # (B, T, N, D)


class TemporalSiglipEncoder(SiglipEncoder):
    """Drop-in replacement for SiglipEncoder implementing space-time separable attention.

    Every `temporal_attn_every_n_layers`th layer l (0-indexed, l % n == 0) is modified as in
    the paper's Appendix C:

        z_hat = z + e(t)                                   # e(0) = 0
        z_hat = z_hat + Attn_time(LN1(z_hat))              # causal, same patch across t
        z_hat = z_hat + Attn_space(LN1(z_hat))             # bidirectional, within frame
        out   = z_hat + MLP(LN2(z_hat))

    i.e. Eq. (3)'s composition alpha_space[alpha_time[z_hat]] (temporal first, as in
    TimeSformer's divided space-time attention), with both attention passes reusing the
    layer's own LN1 + SiglipAttention Q/K/V/O weights. All other layers are stock.

    Past-timestep tokens are dropped right after the last temporal layer: later layers are
    spatial-only, so past frames can no longer influence the current frame and carrying
    them further would be wasted compute. The encoder thus returns only current-frame
    tokens, matching the token count of a single-frame ViT.

    `__init__` is inherited unchanged apart from a non-persistent offsets buffer and a plain
    int, so `self.layers` holds the exact same `SiglipEncoderLayer` instances with identical
    parameter names/shapes as stock -- zero new parameters.
    """

    def __init__(self, config, temporal_attn_every_n_layers: int = 4):
        super().__init__(config)
        self.temporal_attn_every_n_layers = temporal_attn_every_n_layers
        frame_offsets = list(getattr(config, "video_frame_offsets", [0.0]))
        assert frame_offsets == sorted(frame_offsets) and frame_offsets[-1] == 0.0, (
            f"video_frame_offsets must be ascending and end in 0 (current frame last) -- got {frame_offsets}"
        )
        self.register_buffer(
            "video_frame_offsets", torch.tensor(frame_offsets, dtype=torch.float32), persistent=False
        )
        temporal_layers = [
            i for i in range(config.num_hidden_layers) if i % temporal_attn_every_n_layers == 0
        ]
        self.last_temporal_layer = temporal_layers[-1]

    def _space_time_layer(self, layer, hidden_states: torch.Tensor, temporal_pe: torch.Tensor):
        """One modified layer. hidden_states: (B, T, N, C); temporal_pe: (T, C)."""
        b, t, n, c = hidden_states.shape

        # z_hat = z + e(t)
        hidden_states = hidden_states + temporal_pe[None, :, None, :]

        # temporal attention: one length-T sequence per (batch, patch), causal.
        residual = hidden_states
        x = layer.layer_norm1(hidden_states)
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        x = apply_temporal_attention(layer.self_attn, x)
        hidden_states = residual + x.reshape(b, n, t, c).permute(0, 2, 1, 3)

        # spatial attention: standard per-frame block 1, unchanged.
        hidden_states = hidden_states.reshape(b * t, n, c)
        residual = hidden_states
        x = layer.layer_norm1(hidden_states)
        x, _ = layer.self_attn(x, None, output_attentions=False)
        hidden_states = residual + x

        # MLP: standard block 2, unchanged.
        residual = hidden_states
        x = layer.mlp(layer.layer_norm2(hidden_states))
        hidden_states = residual + x
        return hidden_states.view(b, t, n, c)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask=None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        if inputs_embeds.dim() == 3 or inputs_embeds.shape[1] == 1:
            # Single-frame path -- delegate to the literal stock implementation (exact K=1
            # reduction guarantee). Note attention over a length-1 sequence is NOT an
            # identity (out_proj(v_proj(x)) is a nonzero learned map), so T==1 must skip
            # the temporal pass explicitly rather than rely on e(0)=0.
            if inputs_embeds.dim() == 4:
                inputs_embeds = inputs_embeds[:, 0]
            return super().forward(inputs_embeds, attention_mask, output_attentions, output_hidden_states)

        assert inputs_embeds.dim() == 4, f"expected (B,N,C) or (B,T,N,C), got {tuple(inputs_embeds.shape)}"
        assert attention_mask is None, "attention_mask is not supported in the multi-frame video-memory path"
        assert not output_attentions, "output_attentions is not supported in the multi-frame video-memory path"
        b, t, n, c = inputs_embeds.shape
        assert t == self.video_frame_offsets.shape[0], (
            f"got T={t} frames but video_frame_offsets has {self.video_frame_offsets.shape[0]} entries"
        )
        output_hidden_states = bool(
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        encoder_states = () if output_hidden_states else None

        temporal_pe = sinusoidal_temporal_embedding(
            self.video_frame_offsets, c, inputs_embeds.dtype, inputs_embeds.device
        )  # (T, C), zero at the current-frame row

        hidden_states = inputs_embeds  # (B, T, N, C) up to the last temporal layer, then (B, N, C)
        for layer_idx, layer in enumerate(self.layers):
            is_video = layer_idx <= self.last_temporal_layer
            if output_hidden_states:
                # only current-frame states are exposed (select_layer consumers expect (B, N, C))
                encoder_states = encoder_states + ((hidden_states[:, -1] if is_video else hidden_states),)

            if layer_idx % self.temporal_attn_every_n_layers == 0:
                args = (layer, hidden_states, temporal_pe)
                fn = self._space_time_layer
            elif is_video:
                args = (hidden_states.reshape(b * t, n, c), None)
                fn = lambda x, m, layer=layer: layer(x, m)[0].view(b, t, n, c)  # noqa: E731
            else:
                args = (hidden_states, None)
                fn = lambda x, m, layer=layer: layer(x, m)[0]  # noqa: E731

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(fn, *args)
            else:
                hidden_states = fn(*args)

            if layer_idx == self.last_temporal_layer:
                # drop past-timestep tokens: (B, T, N, C) -> (B, N, C), current frame only
                hidden_states = hidden_states[:, -1]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=None)


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
