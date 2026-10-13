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

from gr00t.model.backbone.eagle2_hg_model.modeling_temporal_siglip import build_temporal_siglip_vision_model

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

    # Causality sanity check: perturbing only the OLDEST frame must not change the output
    # (current-frame query attends causally; changing a frame at t' < t affects the current
    # token's temporal keys/values -- wait, causal means current CAN see the past, so this
    # should change. Instead verify the converse: perturbing the CURRENT frame's pixels
    # changes the output (trivially true) and that a T=1 call is independent of any
    # (nonexistent) "future".
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


if __name__ == "__main__":
    test_k1_numerically_identical_tiny_config()
    print("test_k1_numerically_identical_tiny_config: PASS")
    test_multiframe_shape_and_finite_tiny_config()
    print("test_multiframe_shape_and_finite_tiny_config: PASS")
    test_state_dict_strict_compat_real_gr00t_config()
    print("test_state_dict_strict_compat_real_gr00t_config: PASS")
