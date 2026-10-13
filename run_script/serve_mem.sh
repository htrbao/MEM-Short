#!/bin/bash
# Serves a MEM-short (video-memory) GR00T checkpoint over ZMQ, for either libero-mem's
# eval clients or RMBench's policy/GR00T/deploy_policy.py.
#
# No inference_service.py changes were needed: a video-memory checkpoint's saved
# config.json already carries backbone_cfg.video_frame_offsets/temporal_attn_every_n_layers
# (set during training by scripts/gr00t_finetune.py's Stage 2.5 block), so
# GR00T_N1_5.from_pretrained -> EagleBackbone(**backbone_cfg) automatically rebuilds the
# temporal vision tower on load -- this script just points inference_service.py at a
# checkpoint dir and the matching DATA_CONFIG name.
#
# Usage: MODEL_PATH=<checkpoint_dir> DATA_CONFIG=libero_mem_video_memory|rmbench_video_memory \
#        bash run_script/serve_mem.sh
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/mnt/data/sftp/data/baoht9/}"
export PYTHONWARNINGS="ignore"

: "${MODEL_PATH:?Set MODEL_PATH to a trained checkpoint dir (see train_mem_*.sh RUN_DIR)}"
DATA_CONFIG="${DATA_CONFIG:-libero_mem_video_memory}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
PORT="${PORT:-5555}"
HOST="${HOST:-0.0.0.0}"

python scripts/inference_service.py \
  --model-path "$MODEL_PATH" \
  --data-config "$DATA_CONFIG" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --denoising-steps "$DENOISING_STEPS" \
  --port "$PORT" \
  --host "$HOST" \
  --server \
  "$@"
