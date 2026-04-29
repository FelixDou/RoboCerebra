#!/usr/bin/env python3
"""Audit whether eval-time observations match training-time observations.

The LeRobot conversion stores observations generated from per-step HDF5 files.
During evaluation, PI0 observes images/state rendered live from the benchmark
environment. This script resets the eval env to the same stored states and
compares:

  1. stored HDF5 image/state used for training
  2. live eval image/state from the same simulator state
  3. optional PI predictions from both inputs

If offline LeRobot predictions look reasonable but evaluation rollouts look
random, this localizes whether the policy input distribution changed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from policy_adapter import initialize_policy, predict_policy_actions, reset_policy_state
from replay_gt_actions import parse_step_intervals, set_env_state
from task_runner import setup_task_environment
from utils import prepare_observation, process_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocerebra_root", required=True, help="Path to RoboCerebraBench.")
    parser.add_argument("--hdf5_root", required=True, help="Converted HDF5 root containing per_step/.")
    parser.add_argument("--task_type", default="Ideal")
    parser.add_argument("--case_name", default="case5")
    parser.add_argument("--step_index", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="Optional PI0/PI05 checkpoint for prediction comparison.")
    parser.add_argument("--model_family", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--pi_action_seed", type=int, default=None, help="Optional fixed seed before PI select_action.")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def resolve_per_step_hdf5(hdf5_root: Path, case_name: str, step_desc: str) -> Path:
    per_step_root = hdf5_root / "per_step" if (hdf5_root / "per_step").is_dir() else hdf5_root
    step_name = step_desc.replace(" ", "_")
    step_dir = per_step_root / case_name / step_name
    if not step_dir.is_dir():
        raise FileNotFoundError(f"Could not find per-step directory: {step_dir}")
    hdf5_files = sorted(step_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {step_dir}")
    return hdf5_files[0]


def load_hdf5_episode(hdf5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data"]["demo_0"]
        obs = demo["obs"]
        ee_states = np.asarray(obs["ee_states"][()], dtype=np.float32)
        gripper_states = np.asarray(obs["gripper_states"][()], dtype=np.float32)
        return {
            "states": np.asarray(demo["states"][()]),
            "actions": np.asarray(demo["actions"][()], dtype=np.float32),
            "image": np.asarray(obs["agentview_rgb"][()], dtype=np.uint8),
            "wrist_image": np.asarray(obs["eye_in_hand_rgb"][()], dtype=np.uint8),
            "state": np.asarray(np.concatenate((ee_states, gripper_states), axis=-1), dtype=np.float32),
        }


def stored_observation(episode: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {
        "full_image": np.ascontiguousarray(episode["image"][index, ::-1, ::-1, :], dtype=np.uint8),
        "wrist_image": np.ascontiguousarray(episode["wrist_image"][index, ::-1, ::-1, :], dtype=np.uint8),
        "state": np.asarray(episode["state"][index], dtype=np.float32),
    }


def sample_indices(length: int, max_samples: int, stride: int | None, seed: int) -> list[int]:
    if length <= 0:
        return []
    if stride is not None and stride > 0:
        return list(range(0, length, stride))[:max_samples]
    count = min(max_samples, length)
    if count == length:
        return list(range(length))
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(length, size=count, replace=False))


def image_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    return float(np.mean(np.abs(diff))), float(np.max(np.abs(diff)))


def vector_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    return float(np.mean(np.abs(diff))), float(np.max(np.abs(diff)))


def predict_one(cfg: SimpleNamespace, policy_runtime: Any, observation: dict[str, np.ndarray], task: str) -> np.ndarray:
    reset_policy_state(policy_runtime)
    actions = predict_policy_actions(cfg, policy_runtime, observation, task)
    if not actions:
        raise RuntimeError("Policy returned no actions.")
    return np.asarray(process_action(actions[0], cfg.model_family), dtype=np.float32).reshape(-1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.robocerebra_root).expanduser().resolve()
    task_dir = root / args.task_type / args.case_name
    intervals = parse_step_intervals(task_dir / "task_description.txt")
    if args.step_index < 0 or args.step_index >= len(intervals):
        raise ValueError(f"--step_index {args.step_index} out of range for {len(intervals)} intervals")
    task_desc = intervals[args.step_index][0]

    hdf5_path = resolve_per_step_hdf5(Path(args.hdf5_root).expanduser().resolve(), args.case_name, task_desc)
    episode = load_hdf5_episode(hdf5_path)
    indices = sample_indices(len(episode["actions"]), args.max_samples, args.stride, args.seed)

    env, _, error = setup_task_environment(task_dir)
    if error or env is None:
        raise RuntimeError(error or f"Failed to create environment for {task_dir}")

    cfg = None
    policy_runtime = None
    if args.checkpoint:
        cfg = SimpleNamespace(
            model_family=args.model_family,
            pretrained_checkpoint=str(Path(args.checkpoint).expanduser()),
            pi_action_seed=args.pi_action_seed,
        )
        policy_runtime = initialize_policy(cfg)

    rows: list[dict[str, Any]] = []
    try:
        for sample_idx, frame_index in enumerate(indices):
            set_env_state(env, episode["states"][frame_index])
            env_obs, _ = prepare_observation(env._get_observations(), resize_size=None)
            train_obs = stored_observation(episode, frame_index)
            gt_action = episode["actions"][frame_index]

            image_mae, image_max = image_metrics(train_obs["full_image"], env_obs["full_image"])
            wrist_mae, wrist_max = image_metrics(train_obs["wrist_image"], env_obs["wrist_image"])
            state_mae, state_max = vector_metrics(train_obs["state"], env_obs["state"])

            row: dict[str, Any] = {
                "sample_idx": sample_idx,
                "frame_index": int(frame_index),
                "hdf5_path": str(hdf5_path),
                "task": task_desc,
                "image_mae": image_mae,
                "image_max_abs": image_max,
                "wrist_image_mae": wrist_mae,
                "wrist_image_max_abs": wrist_max,
                "state_mae": state_mae,
                "state_max_abs": state_max,
                "train_state": json_cell(train_obs["state"].round(6).tolist()),
                "env_state": json_cell(env_obs["state"].round(6).tolist()),
                "gt_action": json_cell(gt_action.round(6).tolist()),
            }

            if cfg is not None and policy_runtime is not None:
                train_pred = predict_one(cfg, policy_runtime, train_obs, task_desc)
                env_pred = predict_one(cfg, policy_runtime, env_obs, task_desc)
                train_diff = train_pred - gt_action
                env_diff = env_pred - gt_action
                pred_diff = env_pred - train_pred
                row.update(
                    {
                        "train_pred_action_mae": float(np.mean(np.abs(train_diff))),
                        "env_pred_action_mae": float(np.mean(np.abs(env_diff))),
                        "env_vs_train_pred_mae": float(np.mean(np.abs(pred_diff))),
                        "train_pred_action": json_cell(train_pred.round(6).tolist()),
                        "env_pred_action": json_cell(env_pred.round(6).tolist()),
                        "env_vs_train_pred_diff": json_cell(pred_diff.round(6).tolist()),
                    }
                )

            rows.append(row)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    write_csv(Path(args.output_csv).expanduser(), rows)
    summary = {
        "samples": len(rows),
        "hdf5_path": str(hdf5_path),
        "image_mae_mean": float(np.mean([row["image_mae"] for row in rows])) if rows else "",
        "image_max_abs_max": float(np.max([row["image_max_abs"] for row in rows])) if rows else "",
        "wrist_image_mae_mean": float(np.mean([row["wrist_image_mae"] for row in rows])) if rows else "",
        "state_mae_mean": float(np.mean([row["state_mae"] for row in rows])) if rows else "",
        "state_max_abs_max": float(np.max([row["state_max_abs"] for row in rows])) if rows else "",
    }
    if rows and "train_pred_action_mae" in rows[0]:
        summary.update(
            {
                "train_pred_action_mae_mean": float(np.mean([row["train_pred_action_mae"] for row in rows])),
                "env_pred_action_mae_mean": float(np.mean([row["env_pred_action_mae"] for row in rows])),
                "env_vs_train_pred_mae_mean": float(np.mean([row["env_vs_train_pred_mae"] for row in rows])),
            }
        )

    print("EVAL_INPUT_ALIGNMENT_AUDIT_COMPLETE")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  output_csv={Path(args.output_csv).expanduser()}")


if __name__ == "__main__":
    main()
