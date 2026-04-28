#!/usr/bin/env python3
"""Audit PI predictions over every frame of one converted per-step HDF5 episode.

This is stricter than random dataset sampling. It targets one exact step, uses
the stored training observations from the converted HDF5 file, predicts one
action per frame with the same evaluation adapter, and reports:

  - per-dimension action error and cosine
  - temporal shift scores, to catch off-by-k action target bugs
  - simple per-dimension scale fits, to catch action scaling bugs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from audit_eval_input_alignment import (
    json_cell,
    load_hdf5_episode,
    resolve_per_step_hdf5,
    sample_indices,
    stored_observation,
)
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
    parser.add_argument("--model_family", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Cap frames to evaluate.")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_shift", type=int, default=30, help="Evaluate pred[t] vs gt[t+shift].")
    parser.add_argument("--first_n", type=int, default=20, help="Report summary over the first N sampled frames.")
    parser.add_argument("--worst_k", type=int, default=10, help="Report the K highest-error sampled frames.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--shift_csv", required=True)
    return parser.parse_args()


def cosine_values(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1),
        denom,
        out=np.zeros(len(left), dtype=np.float32),
        where=denom > 1e-8,
    )


def error_summary(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    diff = pred - gt
    cosine = cosine_values(pred, gt)
    return {
        "frames": int(len(pred)),
        "action_mae_mean": float(np.mean(np.abs(diff))),
        "action_mae_median": float(np.median(np.mean(np.abs(diff), axis=1))),
        "action_max_abs": float(np.max(np.abs(diff))),
        "per_dim_mae": np.mean(np.abs(diff), axis=0).astype(float).tolist(),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "pred_norm_mean": float(np.mean(np.linalg.norm(pred, axis=1))),
        "gt_norm_mean": float(np.mean(np.linalg.norm(gt, axis=1))),
        "gripper_mae": float(np.mean(np.abs(diff[:, -1]))),
    }


def fit_per_dim_scale(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denom = np.sum(pred * pred, axis=0)
    scale = np.divide(
        np.sum(pred * gt, axis=0),
        denom,
        out=np.zeros(pred.shape[1], dtype=np.float32),
        where=denom > 1e-8,
    )
    scaled_pred = pred * scale[None, :]
    return scale.astype(np.float32), np.mean(np.abs(scaled_pred - gt), axis=0).astype(np.float32)


def frame_error_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": int(row["frame_index"]),
            "action_mae": float(row["action_mae"]),
            "action_max_abs": float(row["action_max_abs"]),
            "cosine": float(row["cosine"]),
            "pred_norm": float(row["pred_norm"]),
            "gt_norm": float(row["gt_norm"]),
        }
        for row in sorted(rows, key=lambda item: float(item["action_mae"]), reverse=True)[: max(0, limit)]
    ]


def shift_summaries(pred: np.ndarray, gt: np.ndarray, max_shift: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            pred_slice = pred[-shift:]
            gt_slice = gt[: len(pred_slice)]
        elif shift > 0:
            pred_slice = pred[: len(pred) - shift]
            gt_slice = gt[shift:]
        else:
            pred_slice = pred
            gt_slice = gt
        if len(pred_slice) == 0:
            continue
        summary = error_summary(pred_slice, gt_slice)
        rows.append(
            {
                "shift": shift,
                "frames": summary["frames"],
                "action_mae_mean": summary["action_mae_mean"],
                "cosine_mean": summary["cosine_mean"],
                "per_dim_mae": json_cell(summary["per_dim_mae"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def predict_episode(args: argparse.Namespace, episode: dict[str, np.ndarray], task_desc: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cfg = SimpleNamespace(model_family=args.model_family, pretrained_checkpoint=str(Path(args.checkpoint).expanduser()))
    policy_runtime = initialize_policy(cfg)

    indices = sample_indices(len(episode["actions"]), args.max_samples, args.stride, args.seed)
    predictions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for sample_idx, frame_index in enumerate(indices):
        observation = stored_observation(episode, frame_index)
        gt_action = np.asarray(episode["actions"][frame_index], dtype=np.float32).reshape(-1)
        reset_policy_state(policy_runtime)
        pred_actions = predict_policy_actions(cfg, policy_runtime, observation, task_desc)
        if not pred_actions:
            raise RuntimeError("Policy returned no actions.")
        pred_action = np.asarray(process_action(pred_actions[0], cfg.model_family), dtype=np.float32).reshape(-1)
        diff = pred_action - gt_action
        cosine = cosine_values(pred_action[None, :], gt_action[None, :])[0]

        predictions.append(pred_action)
        gt_actions.append(gt_action)
        rows.append(
            {
                "sample_idx": sample_idx,
                "frame_index": int(frame_index),
                "task": task_desc,
                "action_mae": float(np.mean(np.abs(diff))),
                "action_max_abs": float(np.max(np.abs(diff))),
                "cosine": float(cosine),
                "pred_norm": float(np.linalg.norm(pred_action)),
                "gt_norm": float(np.linalg.norm(gt_action)),
                "gripper_abs_error": float(abs(diff[-1])),
                "pred_action": json_cell(pred_action.round(6).tolist()),
                "gt_action": json_cell(gt_action.round(6).tolist()),
                "diff": json_cell(diff.round(6).tolist()),
            }
        )

    return np.stack(predictions), np.stack(gt_actions), rows


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
    pred, gt, rows = predict_episode(args, episode, task_desc)
    shifts = shift_summaries(pred, gt, args.max_shift)
    best_shift = min(shifts, key=lambda row: row["action_mae_mean"]) if shifts else {}
    scale, scaled_per_dim_mae = fit_per_dim_scale(pred, gt)
    summary = error_summary(pred, gt)
    first_count = min(max(0, args.first_n), len(pred))
    first_summary = error_summary(pred[:first_count], gt[:first_count]) if first_count else {}
    summary.update(
        {
            "hdf5_path": str(hdf5_path),
            "task": task_desc,
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "first_n": first_count,
            "first_n_action_mae_mean": first_summary.get("action_mae_mean", ""),
            "first_n_action_mae_median": first_summary.get("action_mae_median", ""),
            "first_n_cosine_mean": first_summary.get("cosine_mean", ""),
            "first_n_per_dim_mae": first_summary.get("per_dim_mae", ""),
            "worst_frames": frame_error_rows(rows, args.worst_k),
            "best_shift": best_shift.get("shift", ""),
            "best_shift_action_mae_mean": best_shift.get("action_mae_mean", ""),
            "best_shift_cosine_mean": best_shift.get("cosine_mean", ""),
            "per_dim_scale_fit": scale.astype(float).tolist(),
            "scaled_per_dim_mae": scaled_per_dim_mae.astype(float).tolist(),
        }
    )

    write_csv(Path(args.output_csv).expanduser(), rows)
    write_csv(Path(args.shift_csv).expanduser(), shifts)
    print("PI0_HDF5_EPISODE_PREDICTION_AUDIT_COMPLETE")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  output_csv={Path(args.output_csv).expanduser()}")
    print(f"  shift_csv={Path(args.shift_csv).expanduser()}")


if __name__ == "__main__":
    main()
