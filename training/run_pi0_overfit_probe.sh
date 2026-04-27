#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run a tiny PI0 overfit probe on RoboCerebra RLDS data, then evaluate selected checkpoints.

This is intended to diagnose code/data plumbing. If PI0 cannot visibly improve on a tiny
subset with fixed eval cases, suspect action/camera/proprio/chunking bugs before doing
longer training.

Required environment:
  ROBOCEREBRA_DATA_ROOT   Root containing RoboCerebra_trainset_*_rlds directories.
  BENCH_ROOT              RoboCerebraBench root.

Common optional environment:
  OUT_ROOT                Output root. Default: /gs/bs/tga-shinoda/felid/outputs
  EVAL_LOG_DIR            Eval log dir. Default: /gs/bs/tga-shinoda/felid/eval_logs
  EVAL_ROLLOUT_DIR        Eval rollout dir. Default: /gs/bs/tga-shinoda/felid/eval_rollouts
  LOG_DIR                 Training log dir. Default: /gs/bs/tga-shinoda/felid/logs
  PI0_PRETRAINED          Default: lerobot/pi0_libero_finetuned_v044
  MAX_EPISODES            Default: 10
  STEPS                   Default: 3000
  SAVE_FREQ               Default: 1000
  BATCH_SIZE              Default: 4
  CUDA_VISIBLE_DEVICES    Default: 0
  EVAL_TASK_TYPES         Default: ["Ideal"]
  EVAL_CASE_REGEX         Optional regex for case names. Default: empty
  EVAL_MAX_TASKS_PER_TYPE Default: 1
  EVAL_TRIALS             Default: 1
  WANDB_ENABLE            Default: false
  WANDB_ENTITY            Default: felixdoublet
  WANDB_PROJECT           Default: robocerebra-pi0-overfit

Example:
  ROBOCEREBRA_DATA_ROOT=/gs/bs/tga-shinoda/felid/robocerebra_data \
  BENCH_ROOT=/gs/bs/tga-shinoda/felid/robocerebra_data/RoboCerebraBench \
  CUDA_VISIBLE_DEVICES=0 MAX_EPISODES=10 STEPS=3000 SAVE_FREQ=1000 \
  bash training/run_pi0_overfit_probe.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${ROBOCEREBRA_DATA_ROOT:?Set ROBOCEREBRA_DATA_ROOT to the directory containing RoboCerebra_trainset_*_rlds.}"
: "${BENCH_ROOT:?Set BENCH_ROOT to the RoboCerebraBench directory.}"

OUT_ROOT="${OUT_ROOT:-/gs/bs/tga-shinoda/felid/outputs}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/gs/bs/tga-shinoda/felid/eval_logs}"
EVAL_ROLLOUT_DIR="${EVAL_ROLLOUT_DIR:-/gs/bs/tga-shinoda/felid/eval_rollouts}"
LOG_DIR="${LOG_DIR:-/gs/bs/tga-shinoda/felid/logs}"
PI0_PRETRAINED="${PI0_PRETRAINED:-lerobot/pi0_libero_finetuned_v044}"
MAX_EPISODES="${MAX_EPISODES:-10}"
STEPS="${STEPS:-3000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EVAL_TASK_TYPES="${EVAL_TASK_TYPES:-[\"Ideal\"]}"
EVAL_CASE_REGEX="${EVAL_CASE_REGEX:-}"
EVAL_MAX_TASKS_PER_TYPE="${EVAL_MAX_TASKS_PER_TYPE:-1}"
EVAL_TRIALS="${EVAL_TRIALS:-1}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-felixdoublet}"
WANDB_PROJECT="${WANDB_PROJECT:-robocerebra-pi0-overfit}"
wandb_flag="--no-wandb"
if [[ "$WANDB_ENABLE" == "true" ]]; then
  wandb_flag="--wandb"
fi

export CUDA_VISIBLE_DEVICES
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-/gs/bs/tga-shinoda/felid/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_TF=1

mkdir -p "$OUT_ROOT" "$EVAL_LOG_DIR" "$EVAL_ROLLOUT_DIR" "$LOG_DIR"

RLDS_DIR="${ROBOCEREBRA_DATA_ROOT}/RoboCerebra_trainset_coffee_table_p1p2_rlds/homerobo_trainset_p1p2"
if [[ ! -d "$RLDS_DIR" ]]; then
  echo "Missing RLDS directory: $RLDS_DIR" >&2
  exit 1
fi

JOB="${JOB:-robocerebra_pi0_overfit_${MAX_EPISODES}eps_${STEPS}steps_$(date +%Y%m%d_%H%M%S)}"
TRAIN_LOG="${LOG_DIR}/${JOB}.log"
RUN_OUT="${OUT_ROOT}/${JOB}"

echo "============================================================"
echo "PI0 overfit probe"
echo "JOB=$JOB"
echo "RLDS_DIR=$RLDS_DIR"
echo "MAX_EPISODES=$MAX_EPISODES"
echo "STEPS=$STEPS"
echo "SAVE_FREQ=$SAVE_FREQ"
echo "RUN_OUT=$RUN_OUT"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

python training/finetune_rlds_policy.py \
  --model_family pi0 \
  --rlds_dir "$RLDS_DIR" \
  --max_episodes "$MAX_EPISODES" \
  --pretrained_path "$PI0_PRETRAINED" \
  --job_name "$JOB" \
  --output_dir "$OUT_ROOT" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --save_freq "$SAVE_FREQ" \
  --no-compile_model \
  --no-gradient_checkpointing \
  "$wandb_flag" \
  --extra_arg "--wandb.entity=${WANDB_ENTITY}" \
  --extra_arg "--wandb.project=${WANDB_PROJECT}" \
  2>&1 | tee "$TRAIN_LOG"

if [[ ! -d "$RUN_OUT/checkpoints" ]]; then
  echo "No checkpoints found under $RUN_OUT/checkpoints" >&2
  exit 1
fi

for ckpt_dir in $(find "$RUN_OUT/checkpoints" -maxdepth 1 -mindepth 1 -type d | sort); do
  step="$(basename "$ckpt_dir")"
  pretrained_model="${ckpt_dir}/pretrained_model"
  if [[ ! -d "$pretrained_model" ]]; then
    continue
  fi

  SUBJOB="${JOB}_eval_${step}"
  EVAL_LOG="${EVAL_LOG_DIR}/${SUBJOB}.log"
  ROLLOUT_DIR="${EVAL_ROLLOUT_DIR}/${SUBJOB}"

  echo "============================================================"
  echo "Evaluating checkpoint step=$step"
  echo "CHECKPOINT=$pretrained_model"
  echo "EVAL_LOG=$EVAL_LOG"
  echo "ROLLOUT_DIR=$ROLLOUT_DIR"
  echo "============================================================"

  eval_args=(
    --model_family pi0
    --pretrained_checkpoint "$pretrained_model"
    --robocerebra_root "$BENCH_ROOT"
    --init_files_root "${BENCH_ROOT}/init_files"
    --task_types "$EVAL_TASK_TYPES"
    --num_trials_per_task "$EVAL_TRIALS"
    --max_tasks_per_type "$EVAL_MAX_TASKS_PER_TYPE"
    --local_log_dir "$EVAL_LOG_DIR"
    --rollout_dir "$ROLLOUT_DIR"
    --run_id_note "$SUBJOB"
  )
  if [[ -n "$EVAL_CASE_REGEX" ]]; then
    eval_args+=(--task_case_regex "$EVAL_CASE_REGEX")
  fi

  (
    cd evaluation
    python eval_openvla.py "${eval_args[@]}"
  ) 2>&1 | tee "$EVAL_LOG"
done

echo "============================================================"
echo "Overfit probe complete"
echo "RUN_OUT=$RUN_OUT"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "============================================================"
