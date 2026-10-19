#!/usr/bin/env bash
# Evaluate a MEM-short checkpoint trained with train_mem_libero_t09_10_video_memory.sh on
# LIBERO-Mem T9 (task_id 8) and T10 (task_id 9).
#
# For each task: serves the checkpoint (run_script/serve_mem.sh) on its own GPU/port, then
# runs libero-mem's evaluate_gr00t_plt client, which keeps per-episode frame/state buffers
# and sends the same window the model was trained on:
#   video.agentview_rgb : (6, 256, 256, 3) at offsets -100..0 steps
#   state.state         : (6, 8)
# The client buffers every executed env step (1 env step = 1 dataset frame @ 20Hz), so the
# offsets are exact for any ACTION_HORIZON; it clamps to the oldest buffered frame at episode
# start, like training's first-frame padding. --strip_task_index sends the dataset's task text
# ("put the cream cheese ...") instead of libero's "9 put the cream cheese ...".
# Verified offline: client windows == training samples, actions match GT (mean L1 0.014).
#
# Usage: MODEL_PATH=<ckpt dir> bash run_script/eval_mem_libero_t09_10.sh
# Env overrides: ACTION_HORIZON (1), TAG, TASK_IDS ("8 9"), GPUS ("0 1", one per task, round-robin),
#                BASE_PORT (5741), NUM_TRIALS (50), SEED (42), DENOISING_STEPS (4)
set -uo pipefail
cd "$(dirname "$0")/.."
GR00T_REPO="$(pwd)"
LIBERO_MEM_REPO=/mnt/data/sftp/data/baoht9/workspace/code/libero-mem

: "${MODEL_PATH:?Set MODEL_PATH to a trained checkpoint dir}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"   # executed steps per query (model predicts 16)
TAG="${TAG:-mem-short-$(basename "$MODEL_PATH")_H${ACTION_HORIZON}}"
TASK_IDS=(${TASK_IDS:-8 9})
GPUS=(${GPUS:-0 1})
BASE_PORT="${BASE_PORT:-5741}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-42}"
export DENOISING_STEPS="${DENOISING_STEPS:-4}"
# Must match LiberoMemT0910VideoMemoryDataConfig.observation_indices.
FRAME_OFFSETS="-100 -80 -60 -40 -20 0"

LOG_DIR="$GR00T_REPO/logs/eval_${TAG}"
mkdir -p "$LOG_DIR"
EXP_DIR="./GR00T_N1.5-MEM/${TAG}"   # relative to libero-mem; videos go to ${EXP_DIR}_saved

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

run_task() {
  local tid=$1 gpu=$2 port=$3
  if ss -ltn | grep -q ":${port}\b"; then
    echo "[task $tid] ERROR: port $port already in use" >&2; return 1
  fi
  CUDA_VISIBLE_DEVICES=$gpu MODEL_PATH="$MODEL_PATH" DATA_CONFIG=libero_mem_t09_10_video_memory \
    PORT=$port HOST=127.0.0.1 bash run_script/serve_mem.sh > "$LOG_DIR/serve_task${tid}.log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 120); do
    ss -ltn | grep -q ":${port}\b" && break
    kill -0 $srv 2>/dev/null || { echo "[task $tid] server died, see $LOG_DIR/serve_task${tid}.log" >&2; return 1; }
    sleep 5
  done
  echo "[$(date)] task $tid: server up on GPU $gpu port $port"

  ( cd "$LIBERO_MEM_REPO" && source .venv/bin/activate && \
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu python -m libero.lifelong.evaluate_gr00t_plt \
      --experiment_dir "$EXP_DIR" --benchmark libero_mem --task_id "$tid" \
      --seed "$SEED" --device_id 0 --save-videos --action_horizon "$ACTION_HORIZON" \
      --num_trials_per_task "$NUM_TRIALS" \
      --datasets_dir /mnt/data/sftp/data/baoht9/workspace/data/LIBERO-Mem \
      --host 127.0.0.1 --port "$port" \
      --frame_offsets $FRAME_OFFSETS --send_state_window --no_landmarks --strip_task_index ) \
    > "$LOG_DIR/eval_task${tid}.log" 2>&1
  local rc=$?
  pkill -P $srv 2>/dev/null; kill $srv 2>/dev/null
  echo "[$(date)] task $tid done (rc=$rc): $(grep -E 'Current task success rate' "$LOG_DIR/eval_task${tid}.log" | tail -1)"
}

for i in "${!TASK_IDS[@]}"; do
  run_task "${TASK_IDS[$i]}" "${GPUS[$((i % ${#GPUS[@]}))]}" "$((BASE_PORT + i))" &
  PIDS+=($!)
done
wait

echo "===== $TAG ====="
for tid in "${TASK_IDS[@]}"; do
  echo "task_id $tid: $(grep -E 'Current task success rate' "$LOG_DIR/eval_task${tid}.log" | tail -1)"
done
echo "logs: $LOG_DIR   videos: $LIBERO_MEM_REPO/${EXP_DIR#./}_saved"
