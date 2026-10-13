# SPDX-License-Identifier: Apache-2.0
"""Stage B gate: MEM-short's multi-frame path must not change the LLM's image-placeholder
token count (only the current frame is used for the chat-template text), and the separately
built eagle_video_pixel_values tensor must have the expected (V, T, C, H, W) shape with
exactly one tile per frame.

Exercises GR00TTransform._apply_vlm_processing directly with the real Eagle tokenizer +
image processor (lightweight: no VLM weights are constructed here).

Run with: .venv/bin/python -m pytest tests/test_stage_b_video_pixel_values.py -v
"""

import numpy as np
import torch

from gr00t.model.transforms import GR00TTransform


def _make_transform() -> GR00TTransform:
    return GR00TTransform(
        max_state_dim=8,
        max_action_dim=8,
        state_horizon=1,
        action_horizon=1,
    )


def _fake_video(num_views: int, num_frames: int, h: int = 224, w: int = 224) -> np.ndarray:
    # [V, T, C, H, W] uint8, matching what _prepare_video hands to _apply_vlm_processing.
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(num_views, num_frames, 3, h, w), dtype=np.uint8)


def test_single_frame_baseline_has_no_video_pixel_values_key():
    transform = _make_transform()
    batch = {"images": _fake_video(num_views=1, num_frames=1), "language": "pick up the cup"}
    out = transform._apply_vlm_processing(batch)
    assert "eagle_video_pixel_values" not in out
    assert "eagle_content" in out


def test_multiframe_token_count_matches_single_frame():
    transform = _make_transform()
    num_views = 2
    lang = "pick up the cup"

    batch_1frame = {"images": _fake_video(num_views=num_views, num_frames=1), "language": lang}
    batch_4frame = {"images": _fake_video(num_views=num_views, num_frames=4), "language": lang}

    out_1 = transform._apply_vlm_processing(batch_1frame)
    out_4 = transform._apply_vlm_processing(batch_4frame)

    # apply_chat_template (inside _apply_vlm_processing) only emits <image-N> PLACEHOLDER
    # tags, not yet the real, count-scaling <IMG_CONTEXT> image tokens -- those are
    # produced by replace_media_placeholder, invoked when the processor is actually
    # CALLED (exactly what collate() does: eagle_processor(text=..., images=..., ...)).
    # Replicate that call here to check what really determines input_ids length.
    eagle_content_1 = out_1["eagle_content"]
    eagle_content_4 = out_4["eagle_content"]
    encoded_1 = transform.eagle_processor(
        text=eagle_content_1["text_list"],
        images=eagle_content_1["image_inputs"],
        return_tensors="pt",
        padding=True,
    )
    encoded_4 = transform.eagle_processor(
        text=eagle_content_4["text_list"],
        images=eagle_content_4["image_inputs"],
        return_tensors="pt",
        padding=True,
    )

    image_token_id = transform.eagle_processor.image_token_id
    count_1 = (encoded_1["input_ids"] == image_token_id).sum().item()
    count_4 = (encoded_4["input_ids"] == image_token_id).sum().item()
    assert count_1 == count_4, f"image token count changed with T: {count_1} (T=1) vs {count_4} (T=4)"
    assert count_1 > 0, "sanity: the chat template must contain image tokens at all"
    assert encoded_1["input_ids"].shape == encoded_4["input_ids"].shape

    # 4-frame case must ALSO carry the separate video pixel tensor; 1-frame must not.
    assert "eagle_video_pixel_values" not in out_1
    assert "eagle_video_pixel_values" in out_4


def test_video_pixel_values_shape_one_tile_per_frame():
    transform = _make_transform()
    num_views, num_frames = 2, 4
    batch = {
        "images": _fake_video(num_views=num_views, num_frames=num_frames),
        "language": "pick up the cup",
    }
    out = transform._apply_vlm_processing(batch)
    vpv = out["eagle_video_pixel_values"]
    assert isinstance(vpv, torch.Tensor)
    assert vpv.shape[0] == num_views
    assert vpv.shape[1] == num_frames
    assert vpv.shape[2] == 3
    assert torch.isfinite(vpv).all()


if __name__ == "__main__":
    test_single_frame_baseline_has_no_video_pixel_values_key()
    print("test_single_frame_baseline_has_no_video_pixel_values_key: PASS")
    test_multiframe_token_count_matches_single_frame()
    print("test_multiframe_token_count_matches_single_frame: PASS")
    test_video_pixel_values_shape_one_tile_per_frame()
    print("test_video_pixel_values_shape_one_tile_per_frame: PASS")
