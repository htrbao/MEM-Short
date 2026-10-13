# SPDX-License-Identifier: Apache-2.0
"""Stage B end-to-end gate: a real forward pass through EagleBackbone with
eagle_video_pixel_values present must run and produce the expected output shape,
matching a T=1 call on the same backbone.

Builds inputs the same way GR00TTransform._apply_vlm_processing + collate() do (real
tokenizer/image-processor, not hand-rolled tensors). Requires a CUDA GPU (random-init
~1.3B-param backbone forward on CPU would be too slow for a quick gate check).

Run with: .venv/bin/python tests/test_stage_b_backbone_forward.py
"""

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.backbone.eagle_backbone import EagleBackbone
from gr00t.model.transforms import GR00TTransform


def _make_transform() -> GR00TTransform:
    return GR00TTransform(max_state_dim=8, max_action_dim=8, state_horizon=1, action_horizon=1)


def _build_eagle_batch(transform: GR00TTransform, num_views: int, num_frames: int, device):
    import numpy as np

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(num_views, num_frames, 3, 224, 224), dtype=np.uint8)
    out = transform._apply_vlm_processing({"images": images, "language": "pick up the cup"})
    eagle_content = out["eagle_content"]
    encoded = transform.eagle_processor(
        text=eagle_content["text_list"],
        images=eagle_content["image_inputs"],
        return_tensors="pt",
        padding=True,
    )
    batch = {f"eagle_{k}": v.to(device) for k, v in encoded.items()}
    if num_frames > 1:
        batch["eagle_video_pixel_values"] = out["eagle_video_pixel_values"].to(device)
    return BatchFeature(data=batch)


def main():
    assert torch.cuda.is_available(), "this gate check requires a CUDA GPU"
    device = "cuda"
    transform = _make_transform()

    print("Building video-memory EagleBackbone (K=4)...")
    bb_video = EagleBackbone(
        tune_llm=False,
        tune_visual=False,
        project_to_dim=None,
        select_layer=12,
        video_frame_offsets=[-3.0, -2.0, -1.0, 0.0],
    ).to(device)
    bb_video.eval()

    num_views = 1
    batch_1 = _build_eagle_batch(transform, num_views=num_views, num_frames=1, device=device)
    batch_4 = _build_eagle_batch(transform, num_views=num_views, num_frames=4, device=device)

    with torch.no_grad():
        out_1 = bb_video.forward(batch_1)
        out_4 = bb_video.forward(batch_4)

    feat_1 = out_1["backbone_features"]
    feat_4 = out_4["backbone_features"]
    print(f"T=1 backbone_features shape: {tuple(feat_1.shape)}")
    print(f"T=4 backbone_features shape: {tuple(feat_4.shape)}")
    assert feat_1.shape == feat_4.shape, "token budget must be unchanged by adding past frames"
    assert torch.isfinite(feat_4).all()
    print("PASS: video-memory backbone forward runs and preserves token budget.")

    # And: a backbone with NO video_frame_offsets (today's stock path) must also work
    # unchanged, confirming the default/no-MEM path isn't broken by these edits.
    print("Building stock (K=1) EagleBackbone for regression check...")
    bb_stock = EagleBackbone(
        tune_llm=False, tune_visual=False, project_to_dim=None, select_layer=12
    ).to(device)
    bb_stock.eval()
    with torch.no_grad():
        out_stock = bb_stock.forward(batch_1)
    print(f"stock backbone_features shape: {tuple(out_stock['backbone_features'].shape)}")
    assert out_stock["backbone_features"].shape == feat_1.shape
    print("PASS: stock (no video memory) backbone forward unaffected.")


if __name__ == "__main__":
    main()
