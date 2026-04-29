#!/usr/bin/env python3
"""Audit whether PI predictions align with stored actions or state deltas.

This diagnostic checks a possible action-representation mismatch. RoboCerebra
stores 7D actions named as deltas, while observations store EEF pose/state. For
one HDF5 per-step episode, this script compares:

  - PI prediction vs stored GT action
  - stored GT action vs observed next-state delta
  - PI prediction vs observed next-state delta

The state-delta comparisons include per-dimension scale fits, because simulator
controller commands and realized EEF deltas can differ by a gain factor.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from audit_eval_input_alignment import load_hdf5_episode, resolve_per_step_hdf5, sample_indices, stored_observation
from audit_pi0_hdf5_episode_predictions import cosine_values, fit_per_dim_scale
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
    parser.add_argument("--pi_action_seed", type=int, default=None, help="Optional fixed seed before PI select_action.")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Cap frames to evaluate.")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worst_k", type=int, default=10)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args()


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def predict_one(cfg: SimpleNamespace, policy_runtime: Any, observation: dict[str, np.ndarray], task: str) -> np.ndarray:
    reset_policy_state(policy_runtime)
    actions = predict_policy_actions(cfg, policy_runtime, observation, task)
    if not actions:
        raise RuntimeError("Policy returned no actions.")
    return np.asarray(process_action(actions[0], cfg.model_family), dtype=np.float32).reshape(-1)


def vector_summary(left: np.ndarray, right: np.ndarray, prefix: str, dims: slice | None = None) -> dict[str, Any]:
    left_slice = left[:, dims] if dims is not None else left
    right_slice = right[:, dims] if dims is not None else right
    diff = left_slice - right_slice
    scale, scaled_per_dim_mae = fit_per_dim_scale(left_slice, right_slice)
    scaled = left_slice * scale[None, :]
    scaled_diff = scaled - right_slice
    cosine = cosine_values(left_slice, right_slice)
    scaled_cosine = cosine_values(scaled, right_slice)
    return {
        f"{prefix}_mae_mean": float(np.mean(np.abs(diff))),
        f"{prefix}_mae_median": float(np.median(np.mean(np.abs(diff), axis=1))),
        f"{prefix}_max_abs": float(np.max(np.abs(diff))),
        f"{prefix}_cosine_mean": float(np.mean(cosine)),
        f"{prefix}_cosine_min": float(np.min(cosine)),
        f"{prefix}_per_dim_mae": np.mean(np.abs(diff), axis=0).astype(float).tolist(),
        f"{prefix}_scale_fit": scale.astype(float).tolist(),
        f"{prefix}_scaled_mae_mean": float(np.mean(np.abs(scaled_diff))),
        f"{prefix}_scaled_cosine_mean": float(np.mean(scaled_cosine)),
        f"{prefix}_scaled_per_dim_mae": scaled_per_dim_mae.astype(float).tolist(),
    }


def gripper_targets(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gripper_mean = np.mean(state[:, 6:8], axis=1)
    gripper_delta = gripper_mean[1:] - gripper_mean[:-1]
    gripper_next = gripper_mean[1:]
    return gripper_delta.astype(np.float32), gripper_next.astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def worst_rows(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: float(row[key]), reverse=True)[: max(0, limit)]
    return [
        {
            "frame_index": row["frame_index"],
            "pred_vs_gt_mae": row["pred_vs_gt_mae"],
            "gt_vs_state_delta6_mae": row["gt_vs_state_delta6_mae"],
            "pred_vs_state_delta6_mae": row["pred_vs_state_delta6_mae"],
            "gt_xyz_vs_state_delta_xyz_cosine": row["gt_xyz_vs_state_delta_xyz_cosine"],
            "pred_xyz_vs_state_delta_xyz_cosine": row["pred_xyz_vs_state_delta_xyz_cosine"],
        }
        for row in selected
    ]


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
    usable_length = max(0, len(episode["actions"]) - 1)
    indices = [index for index in sample_indices(usable_length, args.max_samples, args.stride, args.seed) if index + 1 < len(episode["state"])]

    cfg = SimpleNamespace(
        model_family=args.model_family,
        pretrained_checkpoint=str(Path(args.checkpoint).expanduser()),
        pi_action_seed=args.pi_action_seed,
    )
    policy_runtime = initialize_policy(cfg)

    pred_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    state_deltas6: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    gripper_delta, gripper_next = gripper_targets(episode["state"])
    for sample_idx, frame_index in enumerate(indices):
        observation = stored_observation(episode, frame_index)
        pred = predict_one(cfg, policy_runtime, observation, task_desc)
        gt = np.asarray(episode["actions"][frame_index], dtype=np.float32).reshape(-1)
        state_delta6 = np.asarray(episode["state"][frame_index + 1, :6] - episode["state"][frame_index, :6], dtype=np.float32)
        pred6 = pred[:6]
        gt6 = gt[:6]

        pred_actions.append(pred)
        gt_actions.append(gt)
        state_deltas6.append(state_delta6)

        pred_gt_diff = pred - gt
        gt_state_diff = gt6 - state_delta6
        pred_state_diff = pred6 - state_delta6
        rows.append(
            {
                "sample_idx": sample_idx,
                "frame_index": int(frame_index),
                "task": task_desc,
                "pred_vs_gt_mae": float(np.mean(np.abs(pred_gt_diff))),
                "pred_vs_gt_max_abs": float(np.max(np.abs(pred_gt_diff))),
                "gt_vs_state_delta6_mae": float(np.mean(np.abs(gt_state_diff))),
                "pred_vs_state_delta6_mae": float(np.mean(np.abs(pred_state_diff))),
                "gt_xyz_vs_state_delta_xyz_cosine": float(cosine_values(gt6[None, :3], state_delta6[None, :3])[0]),
                "pred_xyz_vs_state_delta_xyz_cosine": float(cosine_values(pred6[None, :3], state_delta6[None, :3])[0]),
                "gt_rot_vs_state_delta_rot_cosine": float(cosine_values(gt6[None, 3:6], state_delta6[None, 3:6])[0]),
                "pred_rot_vs_state_delta_rot_cosine": float(cosine_values(pred6[None, 3:6], state_delta6[None, 3:6])[0]),
                "gt_gripper": float(gt[-1]),
                "pred_gripper": float(pred[-1]),
                "gripper_delta": float(gripper_delta[frame_index]),
                "gripper_next_mean": float(gripper_next[frame_index]),
                "gt_action": json_cell(gt.round(6).tolist()),
                "pred_action": json_cell(pred.round(6).tolist()),
                "state_delta6": json_cell(state_delta6.round(6).tolist()),
            }
        )

    pred_array = np.stack(pred_actions)
    gt_array = np.stack(gt_actions)
    state_delta6_array = np.stack(state_deltas6)

    summary: dict[str, Any] = {
        "frames": int(len(indices)),
        "hdf5_path": str(hdf5_path),
        "task": task_desc,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "pi_action_seed": args.pi_action_seed,
        "worst_pred_vs_gt_frames": worst_rows(rows, "pred_vs_gt_mae", args.worst_k),
    }
    summary.update(vector_summary(pred_array, gt_array, "pred_vs_gt_action7"))
    summary.update(vector_summary(gt_array[:, :6], state_delta6_array, "gt_action6_vs_state_delta6"))
    summary.update(vector_summary(pred_array[:, :6], state_delta6_array, "pred_action6_vs_state_delta6"))
    summary.update(vector_summary(gt_array[:, :3], state_delta6_array[:, :3], "gt_xyz_vs_state_delta_xyz"))
    summary.update(vector_summary(pred_array[:, :3], state_delta6_array[:, :3], "pred_xyz_vs_state_delta_xyz"))

    output_csv = Path(args.output_csv).expanduser()
    summary_json = Path(args.summary_json).expanduser()
    write_csv(output_csv, rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("PI0_ACTION_REPRESENTATION_AUDIT_COMPLETE")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  output_csv={output_csv}")
    print(f"  summary_json={summary_json}")


if __name__ == "__main__":
    main()
