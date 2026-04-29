#!/usr/bin/env python3
"""Audit PI prediction repeatability on exact HDF5 observations.

This targets a simple failure mode: if repeated calls on the same stored
training observation produce different actions, rollout failures can come from
sampling/noise/action-queue behavior rather than from the dataset itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from audit_eval_input_alignment import load_hdf5_episode, resolve_per_step_hdf5, stored_observation
from audit_pi0_hdf5_episode_predictions import cosine_values
from policy_adapter import initialize_policy, predict_policy_actions, reset_policy_state
from replay_gt_actions import parse_step_intervals
from utils import process_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocerebra_root", required=True, help="Path to RoboCerebraBench.")
    parser.add_argument("--hdf5_root", required=True, help="Converted HDF5 root containing per_step/.")
    parser.add_argument("--checkpoint", required=True, help="PI0/PI05 checkpoint path.")
    parser.add_argument("--task_type", default="Ideal")
    parser.add_argument("--case_name", default="case5")
    parser.add_argument("--step_index", type=int, default=0)
    parser.add_argument("--frames", default="0,1,2,3,4,16,17", help="Comma-separated per-step frame indices.")
    parser.add_argument("--num_repeats", type=int, default=20)
    parser.add_argument("--model_family", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--pi_action_seed", type=int, default=None, help="Optional fixed seed before PI select_action.")
    parser.add_argument(
        "--reset_each_repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset policy state before every repeated prediction.",
    )
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def parse_frames(raw: str) -> list[int]:
    frames = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        frames.append(int(item))
    if not frames:
        raise ValueError("--frames must contain at least one integer frame index.")
    return frames


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def predict_repeats(
    cfg: SimpleNamespace,
    policy_runtime: Any,
    observation: dict[str, np.ndarray],
    task: str,
    num_repeats: int,
    reset_each_repeat: bool,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    reset_policy_state(policy_runtime)
    for _ in range(num_repeats):
        if reset_each_repeat:
            reset_policy_state(policy_runtime)
        pred_actions = predict_policy_actions(cfg, policy_runtime, observation, task)
        if not pred_actions:
            raise RuntimeError("Policy returned no actions.")
        action = process_action(pred_actions[0], cfg.model_family)
        predictions.append(np.asarray(action, dtype=np.float32).reshape(-1))
    return np.stack(predictions)


def summarize_frame(frame_index: int, predictions: np.ndarray, gt_action: np.ndarray) -> dict[str, Any]:
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    diff = predictions - gt_action[None, :]
    mean_diff = mean_pred - gt_action
    cosine = cosine_values(predictions, np.repeat(gt_action[None, :], len(predictions), axis=0))
    pairwise = predictions[:, None, :] - predictions[None, :, :]
    pairwise_l2 = np.linalg.norm(pairwise.reshape(-1, predictions.shape[-1]), axis=1)
    return {
        "frame_index": int(frame_index),
        "num_repeats": int(len(predictions)),
        "gt_action": json_cell(gt_action.round(6).tolist()),
        "mean_pred_action": json_cell(mean_pred.round(6).tolist()),
        "std_pred_action": json_cell(std_pred.round(6).tolist()),
        "first_pred_action": json_cell(predictions[0].round(6).tolist()),
        "mean_action_mae": float(np.mean(np.abs(mean_diff))),
        "mean_action_max_abs": float(np.max(np.abs(mean_diff))),
        "repeat_std_mean": float(np.mean(std_pred)),
        "repeat_std_max": float(np.max(std_pred)),
        "repeat_xyz_std": json_cell(std_pred[:3].round(6).tolist()),
        "repeat_gripper_std": float(std_pred[-1]),
        "per_repeat_mae_mean": float(np.mean(np.mean(np.abs(diff), axis=1))),
        "per_repeat_mae_max": float(np.max(np.mean(np.abs(diff), axis=1))),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "pairwise_l2_mean": float(np.mean(pairwise_l2)),
        "pairwise_l2_max": float(np.max(pairwise_l2)),
    }


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
    frames = parse_frames(args.frames)

    cfg = SimpleNamespace(
        model_family=args.model_family,
        pretrained_checkpoint=str(Path(args.checkpoint).expanduser()),
        pi_action_seed=args.pi_action_seed,
    )
    policy_runtime = initialize_policy(cfg)

    rows: list[dict[str, Any]] = []
    for frame_index in frames:
        if frame_index < 0 or frame_index >= len(episode["actions"]):
            raise ValueError(f"Frame index {frame_index} out of range for episode length {len(episode['actions'])}")
        observation = stored_observation(episode, frame_index)
        gt_action = np.asarray(episode["actions"][frame_index], dtype=np.float32).reshape(-1)
        predictions = predict_repeats(
            cfg,
            policy_runtime,
            observation,
            task_desc,
            num_repeats=args.num_repeats,
            reset_each_repeat=args.reset_each_repeat,
        )
        rows.append(summarize_frame(frame_index, predictions, gt_action))

    write_csv(Path(args.output_csv).expanduser(), rows)
    repeat_std_mean = np.asarray([row["repeat_std_mean"] for row in rows], dtype=np.float32)
    pairwise_l2_mean = np.asarray([row["pairwise_l2_mean"] for row in rows], dtype=np.float32)
    mean_action_mae = np.asarray([row["mean_action_mae"] for row in rows], dtype=np.float32)
    print("PI0_PREDICTION_REPEATABILITY_AUDIT_COMPLETE")
    print(f"  frames={len(rows)}")
    print(f"  hdf5_path={hdf5_path}")
    print(f"  checkpoint={Path(args.checkpoint).expanduser()}")
    print(f"  num_repeats={args.num_repeats}")
    print(f"  reset_each_repeat={args.reset_each_repeat}")
    print(f"  pi_action_seed={args.pi_action_seed}")
    print(f"  repeat_std_mean_avg={float(np.mean(repeat_std_mean))}")
    print(f"  repeat_std_mean_max={float(np.max(repeat_std_mean))}")
    print(f"  pairwise_l2_mean_avg={float(np.mean(pairwise_l2_mean))}")
    print(f"  pairwise_l2_mean_max={float(np.max(pairwise_l2_mean))}")
    print(f"  mean_action_mae_avg={float(np.mean(mean_action_mae))}")
    print(f"  mean_action_mae_max={float(np.max(mean_action_mae))}")
    print(f"  output_csv={Path(args.output_csv).expanduser()}")


if __name__ == "__main__":
    main()
