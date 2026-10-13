#!/bin/bash
# MEM-short training on LIBERO-Mem, tasks 9 & 10 only (the "put the cream cheese in the
# nearest basket, then move the OTHER (now-empty) basket to center" sequential-memory
# task pair) -- the only LIBERO-Mem tasks in scope per user's request, and the only ones
# with existing .pruned_init eval coverage in the libero-mem repo already.
#
# Uses data_links/lerobot-LIBERO-MEM-T9-T10, a locally-built mirror of
# /mnt/data/sftp/data/baoht9/workspace/data/lerobot-LIBERO-MEM-landmark filtered down to
# just the 199 episodes (100 T9 + 99 T10) via meta/episodes.jsonl, with per-episode
# symlinks into data/ and videos/ (NOT a bulk directory symlink) so that
# calculate_dataset_statistics's meta/*/*.parquet glob -- which is NOT filtered by
# episodes.jsonl -- only ever sees these 199 episodes' files. See plan Stage C/D.
#
# Usage: bash run_script/train_mem_libero_video_memory.sh
# Env overrides: MAX_STEPS, SAVE_STEPS, BATCH_SIZE, LR, BASE_MODEL, OUTPUT_ROOT, REPORT_TO,
#                NUM_GPUS
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore"
export HF_HOME="${HF_HOME:-/mnt/data/sftp/data/baoht9/}"

# Defaults to tensorboard so this script never needs a hardcoded wandb secret. Export
# WANDB_API_KEY and WANDB_PROJECT yourself beforehand, then pass REPORT_TO=wandb, if you
# want wandb logging.
REPORT_TO="${REPORT_TO:-tensorboard}"

DATA_CONFIG="libero_mem_video_memory"
MODALITY_PATH="$(pwd)/run_script/modality/libero_mem_video_memory_modality.json"
DATA_1="$(pwd)/data_links/lerobot-LIBERO-MEM-T9-T10"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/sftp/data/baoht9/workspace/checkpoints/MEM/gr00tn1d5-libero_mem_video_memory}"

# Must match LiberoMemVideoMemoryDataConfig.observation_indices length
# (gr00t/experiment/data_config.py): [-60, -40, -20, 0], K=4, 3s lookback @ 20fps.
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-4}"
TEMPORAL_ATTN_EVERY_N="${TEMPORAL_ATTN_EVERY_N:-4}"

# Smoke-test defaults (small/fast, to validate correctness end-to-end before committing
# to a full run) -- see plan Stage F. Raise MAX_STEPS/BATCH_SIZE for a real training run;
# start conservative on BATCH_SIZE since video memory runs NUM_VIDEO_FRAMES x the images
# per step through the ViT's spatial-attention pass.
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
NUM_GPUS="${NUM_GPUS:-1}"

RUN_DIR="$OUTPUT_ROOT-$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RUN_DIR"

cp "$MODALITY_PATH" "$DATA_1/meta/modality.json"

python scripts/gr00t_finetune.py \
  --base-model-path "$BASE_MODEL" \
  --output-dir "$RUN_DIR" \
  --dataset-path "$DATA_1" \
  --data-config "$DATA_CONFIG" \
  --video-backend torchvision_av \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --embodiment-tag new_embodiment \
  --num-gpus "$NUM_GPUS" \
  --report_to "$REPORT_TO" \
  --use-video-memory \
  --num-video-frames "$NUM_VIDEO_FRAMES" \
  --temporal-attn-every-n-layers "$TEMPORAL_ATTN_EVERY_N" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "Run dir: $RUN_DIR"
