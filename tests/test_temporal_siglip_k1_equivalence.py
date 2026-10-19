# SPDX-License-Identifier: Apache-2.0
"""K=1 exact-reduction + basic multi-frame sanity checks for MEM-short's video encoder.

Does NOT require downloading pretrained GR00T weights: the K=1-equivalence property is a
purely architectural guarantee (holds for any weights), and state_dict compatibility with
a stock SiglipVisionModel is checked via random init at GR00T's real vision-tower config
(hidden_size=1152, num_hidden_layers=27, ...), not by fetching real pretrained tensors.

Run with: .venv/bin/python -m pytest tests/test_temporal_siglip_k1_equivalence.py -v
"""

import json
import os

import torch
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from gr00t.model.backbone.eagle2_hg_model.modeling_temporal_siglip import (
    apply_temporal_attention,
    build_temporal_siglip_vision_model,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EAGLE_CONFIG_PATH = os.path.join(
    _REPO_ROOT, "gr00t", "model", "backbone", "eagle2_hg_model", "config.json"
)


def _tiny_config() -> SiglipVisionConfig:
    """Small config for fast CPU numerics (not GR00T's real size)."""
    return SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        patch_size=16,
        image_size=64,
        num_channels=3,
        layer_norm_eps=1e-6,
        attn_implementation="eager",
    )


def _real_gr00t_vision_config() -> SiglipVisionConfig:
    """GR00T-N1.5's actual shipped vision-tower config (hidden_size=1152, 27 layers, ...)."""
    with open(_EAGLE_CONFIG_PATH) as f:
        full_cfg = json.load(f)
    vision_cfg = dict(full_cfg["vision_config"])
    vision_cfg.pop("_attn_implementation_autoset", None)
    vision_cfg["attn_implementation"] = "eager"
    return SiglipVisionConfig(**vision_cfg)


def test_k1_numerically_identical_tiny_config():
    """At T=1, the temporal model's output must be BYTE-IDENTICAL to a stock model with
    the same weights -- not just close -- since both code paths literally delegate to the
    same parent SiglipVisionEmbeddings/SiglipEncoder.forward when the time axis is absent.
    """
    torch.manual_seed(0)
    config = _tiny_config()

    stock = SiglipVisionModel(config)
    stock.eval()

    temporal = build_temporal_siglip_vision_model(config, temporal_attn_every_n_layers=4)
    temporal.eval()

    # Prove "no new learnable parameters": a stock state_dict must load into the temporal
    # model with strict=True.
    missing, unexpected = temporal.load_state_dict(stock.state_dict(), strict=True)
    assert not missing and not unexpected

    pixel_values = torch.randn(2, 3, config.image_size, config.image_size)
    with torch.no_grad():
        stock_out = stock(pixel_values=pixel_values).last_hidden_state
        temporal_out_4d = temporal(pixel_values=pixel_values).last_hidden_state  # (B,C,H,W) stock path
        temporal_out_5d = temporal(
            pixel_values=pixel_values.unsqueeze(1)  # (B,1,C,H,W) -- T=1 multi-frame path
        ).last_hidden_state

    assert torch.equal(stock_out, temporal_out_4d)
    # T=1 in the 5D path also must exactly match: config.video_frame_offsets defaults to
    # [0.0], and T==1 there still routes through the multi-frame branch in the encoder
    # (inputs_embeds.dim()==4), but the encoder's explicit `t > 1` guard skips the
    # temporal-attention block entirely at T==1, making the per-layer computation
    # bit-identical to stock -- so this must be exact equality, not just close.
    assert torch.equal(stock_out, temporal_out_5d)


def test_multiframe_shape_and_finite_tiny_config():
    """T>1: output token count must equal the T=1 case (only current-frame tokens kept),
    and outputs must be finite."""
    torch.manual_seed(0)
    config = _tiny_config()
    config.video_frame_offsets = [-3.0, -2.0, -1.0, 0.0]

    model = build_temporal_siglip_vision_model(config, temporal_attn_every_n_layers=4)
    model.eval()

    b = 2
    pixel_values_1frame = torch.randn(b, 3, config.image_size, config.image_size)
    pixel_values_4frame = torch.randn(b, 4, 3, config.image_size, config.image_size)

    with torch.no_grad():
        out1 = model(pixel_values=pixel_values_1frame).last_hidden_state
        out4 = model(pixel_values=pixel_values_4frame).last_hidden_state

    assert out1.shape == out4.shape, "token count must be unchanged by adding past frames"
    assert torch.isfinite(out4).all()

    # The current frame attends causally to all past frames, so perturbing only the
    # OLDEST frame must change the current-frame output.
    pixel_values_4frame_perturbed = pixel_values_4frame.clone()
    pixel_values_4frame_perturbed[:, 0] += 10.0  # perturb the oldest frame only
    with torch.no_grad():
        out4_perturbed = model(pixel_values=pixel_values_4frame_perturbed).last_hidden_state
    assert not torch.allclose(out4, out4_perturbed), (
        "current-frame output should depend on past frames (that's the whole point of "
        "video memory) -- perturbing the oldest frame had no effect"
    )


def test_state_dict_strict_compat_real_gr00t_config():
    """At GR00T's real vision-tower size (hidden_size=1152, 27 layers, patch=14, image=224),
    a stock SiglipVisionModel's state_dict must still load with strict=True -- this is the
    property that lets a real GR00T-N1.5 checkpoint's pretrained vision-tower weights load
    into the video-memory model unmodified."""
    torch.manual_seed(0)
    config = _real_gr00t_vision_config()

    stock = SiglipVisionModel(config)
    temporal = build_temporal_siglip_vision_model(config, temporal_attn_every_n_layers=4)

    missing, unexpected = temporal.load_state_dict(stock.state_dict(), strict=True)
    assert not missing and not unexpected

    # One real-size forward pass at K=1 to confirm shapes/dtypes are sane end-to-end
    # (not a full numerical-equivalence re-check -- that's covered by the tiny-config test).
    stock.eval()
    temporal.eval()
    pixel_values = torch.randn(1, 3, config.image_size, config.image_size)
    with torch.no_grad():
        stock_out = stock(pixel_values=pixel_values).last_hidden_state
        temporal_out = temporal(pixel_values=pixel_values).last_hidden_state
    assert torch.equal(stock_out, temporal_out)


def _reference_space_time_forward(model, pixel_values, every_n):
    """Naive reference straight from the paper's Appendix C, written independently of the
    encoder under test: keeps ALL frames through ALL layers (no token dropping), explicit
    causal mask, and returns post-layernormed current-frame tokens."""
    vm = model.vision_model
    cfg = vm.config
    b, t = pixel_values.shape[:2]
    z = vm.embeddings.patch_embedding(pixel_values.flatten(0, 1)).flatten(2).transpose(1, 2)
    z = z + vm.embeddings.position_embedding(vm.embeddings.position_ids)
    n, c = z.shape[1:]
    z = z.view(b, t, n, c)
    offsets = torch.tensor(cfg.video_frame_offsets)
    omega = 1.0 / (10000.0 ** (torch.arange(c, dtype=torch.float32) / c))
    e = torch.sin(offsets[:, None] * omega[None])  # (T, C)
    assert torch.all(e[-1] == 0)
    h, d = cfg.num_attention_heads, c // cfg.num_attention_heads
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))

    def mha(attn, x, mask=None):  # x: (..., L, C)
        q = attn.q_proj(x).unflatten(-1, (h, d)).transpose(-3, -2)
        k = attn.k_proj(x).unflatten(-1, (h, d)).transpose(-3, -2)
        v = attn.v_proj(x).unflatten(-1, (h, d)).transpose(-3, -2)
        logits = q @ k.transpose(-1, -2) * d**-0.5
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        out = (logits.softmax(-1) @ v).transpose(-3, -2).flatten(-2)
        return attn.out_proj(out)

    for i, layer in enumerate(vm.encoder.layers):
        if i % every_n == 0:
            z = z + e[None, :, None, :]
            zt = z.transpose(1, 2)  # (B, N, T, C)
            zt = zt + mha(layer.self_attn, layer.layer_norm1(zt), causal)
            z = zt.transpose(1, 2)
        z = z + mha(layer.self_attn, layer.layer_norm1(z))
        z = z + layer.mlp(layer.layer_norm2(z))
    return vm.post_layernorm(z[:, -1])


def test_matches_appendix_c_reference_tiny_config():
    """Multi-frame output must match an independent from-the-equations reference (which
    also verifies that dropping past tokens after the last temporal layer is lossless)."""
    torch.manual_seed(0)
    config = _tiny_config()
    config.num_hidden_layers = 10  # last temporal layer = 8, so layer 9 runs on current frame only
    config.video_frame_offsets = [-30.0, -20.0, -10.0, 0.0]
    model = build_temporal_siglip_vision_model(config, temporal_attn_every_n_layers=4).eval()
    pixel_values = torch.randn(2, 4, 3, config.image_size, config.image_size)
    with torch.no_grad():
        out = model(pixel_values=pixel_values).last_hidden_state
        ref = _reference_space_time_forward(model, pixel_values, every_n=4)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_temporal_attention_is_causal():
    """A frame's temporal-attention output must not depend on any later frame."""
    torch.manual_seed(0)
    config = _tiny_config()
    attn = SiglipVisionModel(config).vision_model.encoder.layers[0].self_attn.eval()
    x = torch.randn(5, 4, config.hidden_size)
    x_perturbed = x.clone()
    x_perturbed[:, -1] += 10.0
    with torch.no_grad():
        out = apply_temporal_attention(attn, x)
        out_perturbed = apply_temporal_attention(attn, x_perturbed)
    torch.testing.assert_close(out[:, :-1], out_perturbed[:, :-1])
    assert not torch.allclose(out[:, -1], out_perturbed[:, -1])


def test_gradient_checkpointing_matches():
    torch.manual_seed(0)
    config = _tiny_config()
    config.video_frame_offsets = [-2.0, -1.0, 0.0]
    model = build_temporal_siglip_vision_model(config, temporal_attn_every_n_layers=4).train()
    pixel_values = torch.randn(2, 3, 3, config.image_size, config.image_size)

    model(pixel_values=pixel_values).last_hidden_state.square().sum().backward()
    grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model(pixel_values=pixel_values).last_hidden_state.square().sum().backward()
    for k, p in model.named_parameters():
        if k in grads:
            torch.testing.assert_close(p.grad, grads[k], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_k1_numerically_identical_tiny_config()
    print("test_k1_numerically_identical_tiny_config: PASS")
    test_multiframe_shape_and_finite_tiny_config()
    print("test_multiframe_shape_and_finite_tiny_config: PASS")
    test_state_dict_strict_compat_real_gr00t_config()
    print("test_state_dict_strict_compat_real_gr00t_config: PASS")
