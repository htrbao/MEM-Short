#!/bin/bash
# MEM-short training on RMBench's Observe&PickUp task -- the only RMBench task in scope
# per user's request. RMBench-Observe_n_Pickup-lerobot is already single-task (one entry
# in meta/tasks.jsonl), so unlike LIBERO-Mem no episode filtering is needed -- the mirror
# at data_links/RMBench-Observe_n_Pickup-video-memory is a plain bulk symlink of the full
# dataset with a MEM-specific (non-landmark) meta/modality.json swapped in.
#
# Usage: bash run_script/train_mem_rmbench_video_memory.sh
# Env overrides: MAX_STEPS, SAVE_STEPS, BATCH_SIZE, LR, BASE_MODEL, OUTPUT_ROOT, REPORT_TO
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore"
export HF_HOME="${HF_HOME:-/mnt/data/sftp/data/baoht9/}"

# Defaults to tensorboard so this script never needs a hardcoded wandb secret. Export
# WANDB_API_KEY and WANDB_PROJECT yourself beforehand, then pass REPORT_TO=wandb, if you
# want wandb logging.
REPORT_TO="${REPORT_TO:-tensorboard}"

DATA_CONFIG="rmbench_video_memory"
MODALITY_PATH="$(pwd)/run_script/modality/rmbench_video_memory_modality.json"
DATA_1="$(pwd)/data_links/RMBench-Observe_n_Pickup-video-memory"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/sftp/data/baoht9/workspace/checkpoints/MEM/gr00tn1d5-rmbench_video_memory}"

# Must match RmBenchVideoMemoryDataConfig.observation_indices length
# (gr00t/experiment/data_config.py): [-120, -80, -40, 0], K=4, 2.4s lookback @ 50fps --
# reuses the exact offsets already proven to work for this dataset by the sibling PLT
# repos' RMBenchObserveNPickupRecurrentDataConfig.
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-4}"
TEMPORAL_ATTN_EVERY_N="${TEMPORAL_ATTN_EVERY_N:-4}"

# Smoke-test defaults -- see plan Stage F. RMBench-Observe_n_Pickup is small (50
# episodes/7473 frames), so even a real run here needs far fewer steps than LIBERO-Mem;
# PLT's own vanilla-RMBench reference run used 6000 steps at bs=32 (~26 epochs).
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"

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
  --num-gpus 1 \
  --report_to "$REPORT_TO" \
  --use-video-memory \
  --num-video-frames "$NUM_VIDEO_FRAMES" \
  --temporal-attn-every-n-layers "$TEMPORAL_ATTN_EVERY_N" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "Run dir: $RUN_DIR"
