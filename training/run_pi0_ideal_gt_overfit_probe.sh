#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Train PI0 on RoboCerebraBench Ideal ground-truth demos, then evaluate on the same Ideal cases.

This is the strict code-plumbing overfit test. It uses benchmark demo.hdf5 actions/states
as the training target, so do not report it as benchmark performance. Use it to answer:
"Can the policy learn and execute the exact GT demonstrations used by evaluation?"

Required environment:
  BENCH_ROOT              RoboCerebraBench root containing Ideal/.

Common optional environment:
  OUT_ROOT                Output root. Default: /gs/bs/tga-shinoda/felid/outputs
  WORK_ROOT               Intermediate conversion root. Default: /gs/bs/tga-shinoda/felid/pi0_ideal_gt_overfit
  EVAL_LOG_DIR            Eval log dir. Default: /gs/bs/tga-shinoda/felid/eval_logs
  EVAL_ROLLOUT_DIR        Eval rollout dir. Default: /gs/bs/tga-shinoda/felid/eval_rollouts
  LOG_DIR                 Log dir. Default: /gs/bs/tga-shinoda/felid/logs
  PI0_PRETRAINED          Default: lerobot/pi0_libero_finetuned_v044
  MAX_CASES               Default: 1
  CASE_REGEX              Optional regex for Ideal case names. Default: empty
  STEPS                   Default: 3000
  SAVE_FREQ               Default: 1000
  BATCH_SIZE              Default: 4
  CUDA_VISIBLE_DEVICES    Default: 0
  EVAL_TRIALS             Default: 1
  STRICT_ORIGINAL_ONLY    Disable texture/distractor variants. Default: true
  WANDB_ENABLE            Default: false
  WANDB_ENTITY            Default: felixdoublet
  WANDB_PROJECT           Default: robocerebra-pi0-overfit

Example:
  BENCH_ROOT=/gs/bs/tga-shinoda/felid/robocerebra_data/RoboCerebraBench \
  MAX_CASES=1 STEPS=3000 SAVE_FREQ=1000 CUDA_VISIBLE_DEVICES=0 \
  bash training/run_pi0_ideal_gt_overfit_probe.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${BENCH_ROOT:?Set BENCH_ROOT to the RoboCerebraBench directory.}"

OUT_ROOT="${OUT_ROOT:-/gs/bs/tga-shinoda/felid/outputs}"
WORK_ROOT="${WORK_ROOT:-/gs/bs/tga-shinoda/felid/pi0_ideal_gt_overfit}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/gs/bs/tga-shinoda/felid/eval_logs}"
EVAL_ROLLOUT_DIR="${EVAL_ROLLOUT_DIR:-/gs/bs/tga-shinoda/felid/eval_rollouts}"
LOG_DIR="${LOG_DIR:-/gs/bs/tga-shinoda/felid/logs}"
PI0_PRETRAINED="${PI0_PRETRAINED:-lerobot/pi0_libero_finetuned_v044}"
MAX_CASES="${MAX_CASES:-1}"
CASE_REGEX="${CASE_REGEX:-}"
STEPS="${STEPS:-3000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EVAL_TRIALS="${EVAL_TRIALS:-1}"
STRICT_ORIGINAL_ONLY="${STRICT_ORIGINAL_ONLY:-true}"
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

mkdir -p "$OUT_ROOT" "$WORK_ROOT" "$EVAL_LOG_DIR" "$EVAL_ROLLOUT_DIR" "$LOG_DIR"

IDEAL_ROOT="${BENCH_ROOT}/Ideal"
if [[ ! -d "$IDEAL_ROOT" ]]; then
  echo "Missing Ideal benchmark directory: $IDEAL_ROOT" >&2
  exit 1
fi

JOB="${JOB:-robocerebra_pi0_ideal_gt_overfit_${MAX_CASES}cases_${STEPS}steps_$(date +%Y%m%d_%H%M%S)}"
HDF5_ROOT="${WORK_ROOT}/${JOB}/converted_hdf5"
LEROBOT_ROOT="${WORK_ROOT}/${JOB}/lerobot_datasets"
DATASET_REPO_ID="robocerebra/${JOB}_dataset"
DATASET_ROOT="${LEROBOT_ROOT}/${DATASET_REPO_ID}"
TRAIN_LOG="${LOG_DIR}/${JOB}.log"
RUN_OUT="${OUT_ROOT}/${JOB}"

echo "============================================================"
echo "PI0 Ideal GT overfit probe"
echo "JOB=$JOB"
echo "IDEAL_ROOT=$IDEAL_ROOT"
echo "MAX_CASES=$MAX_CASES"
echo "CASE_REGEX=$CASE_REGEX"
echo "STRICT_ORIGINAL_ONLY=$STRICT_ORIGINAL_ONLY"
echo "HDF5_ROOT=$HDF5_ROOT"
echo "LEROBOT_ROOT=$LEROBOT_ROOT"
echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "RUN_OUT=$RUN_OUT"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

regen_args=(
  --dataset_name "$JOB"
  --robocerebra_raw_data_dir "$IDEAL_ROOT"
  --robocerebra_target_dir "$HDF5_ROOT"
  --max_cases "$MAX_CASES"
  --overwrite
)
if [[ -n "$CASE_REGEX" ]]; then
  regen_args+=(--case_regex "$CASE_REGEX")
fi
if [[ "$STRICT_ORIGINAL_ONLY" == "true" ]]; then
  regen_args+=(--no_texture_variants --no_distractor_variants)
fi
python rlds_dataset_builder/regenerate_robocerebra_dataset.py "${regen_args[@]}" 2>&1 | tee "$TRAIN_LOG"

convert_args=(
  --robocerebra_hdf5_root "$HDF5_ROOT"
  --repo_id "$DATASET_REPO_ID"
  --root "$LEROBOT_ROOT"
  --overwrite
)
python rlds_dataset_builder/convert_hdf5_to_lerobot.py "${convert_args[@]}" 2>&1 | tee -a "$TRAIN_LOG"

python training/finetune_lerobot_policy.py \
  --model_family pi0 \
  --dataset_repo_id "$DATASET_REPO_ID" \
  --dataset_root "$DATASET_ROOT" \
  --pretrained_path "$PI0_PRETRAINED" \
  --job_name "$JOB" \
  --output_dir "$OUT_ROOT" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --no-compile_model \
  --no-gradient_checkpointing \
  "$wandb_flag" \
  "--extra_arg=--save_freq=${SAVE_FREQ}" \
  "--extra_arg=--policy.push_to_hub=false" \
  "--extra_arg=--wandb.entity=${WANDB_ENTITY}" \
  "--extra_arg=--wandb.project=${WANDB_PROJECT}" \
  2>&1 | tee -a "$TRAIN_LOG"

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

  eval_args=(
    --model_family pi0
    --pretrained_checkpoint "$pretrained_model"
    --robocerebra_root "$BENCH_ROOT"
    --init_files_root "${BENCH_ROOT}/init_files"
    --task_types '["Ideal"]'
    --num_trials_per_task "$EVAL_TRIALS"
    --max_tasks_per_type "$MAX_CASES"
    --local_log_dir "$EVAL_LOG_DIR"
    --rollout_dir "$ROLLOUT_DIR"
    --run_id_note "$SUBJOB"
  )
  if [[ -n "$CASE_REGEX" ]]; then
    eval_args+=(--task_case_regex "$CASE_REGEX")
  fi

  echo "============================================================"
  echo "Evaluating checkpoint step=$step"
  echo "CHECKPOINT=$pretrained_model"
  echo "EVAL_LOG=$EVAL_LOG"
  echo "ROLLOUT_DIR=$ROLLOUT_DIR"
  echo "============================================================"

  (
    cd evaluation
    python eval_openvla.py "${eval_args[@]}"
  ) 2>&1 | tee "$EVAL_LOG"
done

echo "============================================================"
echo "Ideal GT overfit probe complete"
echo "RUN_OUT=$RUN_OUT"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "============================================================"
