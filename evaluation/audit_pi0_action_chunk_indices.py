#!/usr/bin/env python3
"""Audit which PI0 predicted action-chunk index best matches GT actions.

PI-family policies can emit an action chunk, while RoboCerebra evaluation
executes one action at a time. If evaluation extracts the wrong chunk element,
offline first-action metrics and closed-loop behavior can look bad even when a
later element in the predicted chunk is aligned with the demonstration action.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from audit_eval_input_alignment import load_hdf5_episode, resolve_per_step_hdf5, sample_indices, stored_observation
from audit_pi0_hdf5_episode_predictions import cosine_values
from policy_adapter import (
    _build_lerobot_batch,
    _ensure_batch_dim,
    _move_to_device,
    _to_action_sequence,
    initialize_policy,
    reset_policy_state,
)
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
    parser.add_argument("--pi_action_seed", type=int, default=None, help="Optional fixed seed before PI sampling.")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Cap frames to evaluate.")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first_n", type=int, default=30, help="Also summarize the first N sampled frames.")
    parser.add_argument("--top_k", type=int, default=10, help="Print the top K chunk indices by MAE.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args()


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def clone_prediction(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_prediction(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_prediction(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_prediction(item) for item in value)
    return copy.deepcopy(value)


def set_optional_pi_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def action_sequence_to_array(actions: Any, model_family: str) -> np.ndarray:
    sequence = _to_action_sequence(actions)
    processed = [np.asarray(process_action(action, model_family), dtype=np.float32).reshape(-1) for action in sequence]
    if not processed:
        raise RuntimeError("Policy returned an empty action sequence.")
    return np.stack(processed)


def build_processed_batch(policy_runtime: Any, observation: dict[str, Any], task: str) -> dict[str, Any]:
    batch = _build_lerobot_batch(policy_runtime, observation, task)
    if policy_runtime.preprocessor is not None:
        batch = policy_runtime.preprocessor(batch)
    batch = _ensure_batch_dim(batch, policy_runtime.input_features)
    return _move_to_device(batch, policy_runtime.device)


def postprocess_action_object(policy_runtime: Any, actions: Any) -> Any:
    if policy_runtime.postprocessor is None:
        return actions
    return policy_runtime.postprocessor(actions)


def select_action_chunk(args: argparse.Namespace, policy_runtime: Any, batch: dict[str, Any]) -> np.ndarray:
    reset_policy_state(policy_runtime)
    set_optional_pi_seed(args.pi_action_seed)
    with torch.inference_mode():
        actions = policy_runtime.model.select_action(batch)
    actions = postprocess_action_object(policy_runtime, clone_prediction(actions))
    return action_sequence_to_array(actions, args.model_family)


def chunk_error_summary(chunks: list[np.ndarray], gt_actions: np.ndarray, first_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_chunk_len = max(chunk.shape[0] for chunk in chunks)
    rows: list[dict[str, Any]] = []

    for chunk_index in range(max_chunk_len):
        pred_rows = []
        gt_rows = []
        frame_indices = []
        for frame_idx, (chunk, gt_action) in enumerate(zip(chunks, gt_actions, strict=True)):
            if chunk_index >= chunk.shape[0]:
                continue
            pred_rows.append(chunk[chunk_index])
            gt_rows.append(gt_action)
            frame_indices.append(frame_idx)
        if not pred_rows:
            continue

        pred = np.stack(pred_rows)
        gt = np.stack(gt_rows)
        diff = pred - gt
        cosine = cosine_values(pred, gt)
        first_count = min(max(0, first_n), len(pred))
        first_diff = diff[:first_count] if first_count else None
        first_cosine = cosine[:first_count] if first_count else None
        rows.append(
            {
                "chunk_index": chunk_index,
                "frames": int(len(pred)),
                "action_mae_mean": float(np.mean(np.abs(diff))),
                "action_mae_median": float(np.median(np.mean(np.abs(diff), axis=1))),
                "action_max_abs": float(np.max(np.abs(diff))),
                "cosine_mean": float(np.mean(cosine)),
                "cosine_min": float(np.min(cosine)),
                "first_n": int(first_count),
                "first_n_action_mae_mean": float(np.mean(np.abs(first_diff))) if first_diff is not None else "",
                "first_n_cosine_mean": float(np.mean(first_cosine)) if first_cosine is not None else "",
                "per_dim_mae": json_cell(np.mean(np.abs(diff), axis=0).astype(float).tolist()),
                "first_frame_position_indices": json_cell(frame_indices[:5]),
            }
        )

    best_by_mae = min(rows, key=lambda row: row["action_mae_mean"]) if rows else {}
    best_by_first_n = (
        min(rows, key=lambda row: float(row["first_n_action_mae_mean"]))
        if rows and rows[0]["first_n_action_mae_mean"] != ""
        else {}
    )
    summary = {
        "chunk_count": int(max_chunk_len),
        "best_chunk_index": best_by_mae.get("chunk_index", ""),
        "best_chunk_action_mae_mean": best_by_mae.get("action_mae_mean", ""),
        "best_chunk_cosine_mean": best_by_mae.get("cosine_mean", ""),
        "first_action_mae_mean": rows[0]["action_mae_mean"] if rows else "",
        "first_action_cosine_mean": rows[0]["cosine_mean"] if rows else "",
        "best_first_n_chunk_index": best_by_first_n.get("chunk_index", ""),
        "best_first_n_action_mae_mean": best_by_first_n.get("first_n_action_mae_mean", ""),
        "best_first_n_cosine_mean": best_by_first_n.get("first_n_cosine_mean", ""),
    }
    return rows, summary


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

    cfg = SimpleNamespace(
        model_family=args.model_family,
        pretrained_checkpoint=str(Path(args.checkpoint).expanduser()),
        pi_action_seed=args.pi_action_seed,
    )
    policy_runtime = initialize_policy(cfg)

    chunks: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    chunk_lengths: list[int] = []
    for frame_index in indices:
        observation = stored_observation(episode, frame_index)
        gt_action = np.asarray(episode["actions"][frame_index], dtype=np.float32).reshape(-1)
        batch = build_processed_batch(policy_runtime, observation, task_desc)
        chunk = select_action_chunk(args, policy_runtime, batch)
        chunks.append(chunk)
        gt_actions.append(gt_action)
        chunk_lengths.append(int(chunk.shape[0]))

    rows, summary = chunk_error_summary(chunks, np.stack(gt_actions), args.first_n)
    top_rows = sorted(rows, key=lambda row: row["action_mae_mean"])[: max(0, args.top_k)]
    summary.update(
        {
            "frames": int(len(indices)),
            "hdf5_path": str(hdf5_path),
            "task": task_desc,
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "pi_action_seed": args.pi_action_seed,
            "chunk_lengths_min": int(min(chunk_lengths)) if chunk_lengths else 0,
            "chunk_lengths_max": int(max(chunk_lengths)) if chunk_lengths else 0,
            "top_chunks_by_mae": top_rows,
        }
    )

    output_csv = Path(args.output_csv).expanduser()
    summary_json = Path(args.summary_json).expanduser()
    write_csv(output_csv, rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("PI0_ACTION_CHUNK_INDEX_AUDIT_COMPLETE")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  output_csv={output_csv}")
    print(f"  summary_json={summary_json}")


if __name__ == "__main__":
    main()
