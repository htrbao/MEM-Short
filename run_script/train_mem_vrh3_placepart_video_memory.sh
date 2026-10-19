#!/bin/bash
# MEM-short (short-horizon video memory, MEM paper Sec. III-C) on VRH3 part-placing sim.
#
# Dataset: vrh3-part-placing-simulation-landmark (99 episodes, 1 task, 20fps), read through
# a private mirror under data_links/ (own meta/, symlinked data/ + videos/) so other jobs
# rewriting the shared meta/modality.json can't break this run. Never writes into the
# shared dataset directory.
#
# Data config `vrh3_placepart_video_memory`: cam_front only (observation.images.color.front),
# joint-position state only (no velocity/effort), K=6 observations (5 past + current) at a
# 1s stride, with matching proprio history. The vision tower is trained (--tune-visual):
# the temporal attention reuses the ViT's spatial attention weights.
#
# Usage: bash run_script/train_mem_vrh3_placepart_video_memory.sh
# Env overrides: DATASET, BASE_MODEL, OUTPUT_ROOT, MAX_STEPS, SAVE_STEPS, BATCH_SIZE, LR,
#                NUM_GPUS, GRAD_ACCUM, NUM_WORKERS, TEMPORAL_ATTN_EVERY_N, REPORT_TO,
#                WANDB_PROJECT, WANDB_ENTITY
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore"
export HF_HOME="${HF_HOME:-/mnt/data/sftp/data/baoht9/}"

DATASET="${DATASET:-/mnt/data/sftp/data/baoht9/workspace/data/vrh3-part-placing-simulation-landmark}"
DATA_CONFIG="vrh3_placepart_video_memory"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/sftp/data/baoht9/workspace/checkpoints/MEM}"

# Must equal len(Vrh3PlacePartVideoMemoryDataConfig.observation_indices).
NUM_VIDEO_FRAMES=6
TEMPORAL_ATTN_EVERY_N="${TEMPORAL_ATTN_EVERY_N:-4}"

MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"   # per GPU
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# wandb auth comes from ~/.netrc (`wandb login`) or WANDB_API_KEY; REPORT_TO=tensorboard to opt out.
REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-GR00TN1.5 MEMORY}"

# <baseline-name>_<data-config>_<memory config>_<date>
RUN_DIR="$OUTPUT_ROOT/gr00tn1d5_${DATA_CONFIG}_K${NUM_VIDEO_FRAMES}-stride20-every${TEMPORAL_ATTN_EVERY_N}_$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RUN_DIR"

MIRROR="$(pwd)/data_links/$(basename "$DATASET")-mem"
mkdir -p "$MIRROR/meta"
for f in info.json episodes.jsonl episodes_stats.jsonl tasks.jsonl stats.json; do
  [ -f "$DATASET/meta/$f" ] && cp "$DATASET/meta/$f" "$MIRROR/meta/$f"
done
cp "$(pwd)/run_script/modality/vrh3_placepart_video_memory_modality.json" "$MIRROR/meta/modality.json"
ln -sfn "$DATASET/data" "$MIRROR/data"
ln -sfn "$DATASET/videos" "$MIRROR/videos"

python scripts/gr00t_finetune.py \
  --base-model-path "$BASE_MODEL" \
  --output-dir "$RUN_DIR" \
  --dataset-path "$MIRROR" \
  --data-config "$DATA_CONFIG" \
  --video-backend torchvision_av \
  --embodiment-tag new_embodiment \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --num-gpus "$NUM_GPUS" \
  --dataloader-num-workers "$NUM_WORKERS" \
  --report_to "$REPORT_TO" \
  --tune-visual \
  --use-video-memory \
  --num-video-frames "$NUM_VIDEO_FRAMES" \
  --temporal-attn-every-n-layers "$TEMPORAL_ATTN_EVERY_N" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "Run dir: $RUN_DIR"
