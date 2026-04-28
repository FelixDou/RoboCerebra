#!/usr/bin/env python3
"""Audit RoboCerebra HDF5 -> per-step HDF5 -> LeRobot action conversion.

This diagnostic checks whether the training data given to LeRobot preserves the
original benchmark `demo.hdf5` actions after the same no-op filtering and step
splitting used by `regenerate_robocerebra_dataset.py`.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class StepSpec:
    case_name: str
    step_index: int
    step_name: str
    description: str
    old_end_frame: int
    hdf5_path: Path | None


def is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1])


def parse_step_file(txt_path: Path) -> list[tuple[str, int]]:
    steps: list[tuple[str, int]] = []
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        if not lines[idx].startswith("Step:"):
            idx += 1
            continue
        desc = lines[idx].split(":", 1)[1].strip()
        next_idx = idx + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1

        if next_idx < len(lines) and lines[next_idx].lstrip().startswith("["):
            match = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", lines[next_idx].strip())
            if not match:
                raise ValueError(f"{txt_path}: invalid frame interval: {lines[next_idx]}")
            steps.append((desc, int(match.group(2))))
            idx = next_idx + 1
            continue

        try:
            desc, end_frame_str = desc.rsplit(" ", 1)
            steps.append((desc.strip(), int(end_frame_str)))
        except ValueError as exc:
            raise ValueError(f"{txt_path}: could not parse step line: {lines[idx]}") from exc
        idx += 1

    if not steps:
        raise ValueError(f"No steps found in {txt_path}")
    return sorted(steps, key=lambda item: item[1])


def step_name_from_description(description: str) -> str:
    return description.replace(" ", "_")


def load_demo_actions(demo_path: Path) -> np.ndarray:
    with h5py.File(demo_path, "r") as handle:
        return np.asarray(handle["data"]["demo_1"]["actions"][()], dtype=np.float32)


def load_per_step_actions(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as handle:
        return np.asarray(handle["data"]["demo_0"]["actions"][()], dtype=np.float32)


def filtered_action_segments(raw_actions: np.ndarray, steps: list[tuple[str, int]]) -> tuple[list[np.ndarray], list[int]]:
    keep_idx: list[int] = []
    kept_actions: list[np.ndarray] = []
    prev_action = None
    for idx, action in enumerate(raw_actions):
        if is_noop(action, prev_action):
            prev_action = action
            continue
        keep_idx.append(idx)
        kept_actions.append(action)
        prev_action = action

    if not kept_actions:
        return [], keep_idx

    actions = np.asarray(kept_actions, dtype=np.float32)
    new_step_ends = [min(bisect.bisect_right(keep_idx, old_end), len(actions)) for _, old_end in steps]

    segments: list[np.ndarray] = []
    prev_end = 0
    for new_end in new_step_ends:
        if prev_end >= len(actions):
            break
        new_end = min(max(new_end, prev_end), len(actions))
        if new_end > prev_end:
            segments.append(actions[prev_end:new_end])
        prev_end = new_end
    return segments, keep_idx


def discover_step_specs(bench_root: Path, hdf5_root: Path, case_regex: str | None) -> list[StepSpec]:
    per_step_root = hdf5_root / "per_step" if hdf5_root.name != "per_step" else hdf5_root
    if not per_step_root.is_dir():
        raise FileNotFoundError(f"Missing per_step directory: {per_step_root}")

    pattern = re.compile(case_regex) if case_regex else None
    specs: list[StepSpec] = []
    for case_dir in sorted(path for path in (bench_root / "Ideal").iterdir() if path.is_dir()):
        if pattern and not pattern.search(case_dir.name):
            continue
        steps = parse_step_file(case_dir / "task_description.txt")
        for step_idx, (description, old_end_frame) in enumerate(steps):
            step_name = step_name_from_description(description)
            step_dir = per_step_root / case_dir.name / step_name
            hdf5_files = sorted(step_dir.glob("*.hdf5")) if step_dir.is_dir() else []
            if not hdf5_files:
                specs.append(
                    StepSpec(case_dir.name, step_idx, step_name, description, old_end_frame, None)
                )
                continue
            specs.append(
                StepSpec(case_dir.name, step_idx, step_name, description, old_end_frame, hdf5_files[0])
            )
    return specs


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def action_stats(actions: np.ndarray) -> dict[str, Any]:
    if len(actions) == 0:
        return {"frames": 0}
    return {
        "frames": int(len(actions)),
        "shape": list(actions.shape),
        "min": np.min(actions, axis=0).round(6).tolist(),
        "max": np.max(actions, axis=0).round(6).tolist(),
        "mean": np.mean(actions, axis=0).round(6).tolist(),
        "std": np.std(actions, axis=0).round(6).tolist(),
        "gripper_values": sorted(float(x) for x in np.unique(actions[:, -1])),
    }


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def read_lerobot_actions(dataset_root: Path) -> list[np.ndarray]:
    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    if not parquet_files:
        return []
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Reading LeRobot parquet files requires pandas/pyarrow.") from exc

    episodes: list[np.ndarray] = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=["action"])
        actions = np.asarray([np.asarray(value, dtype=np.float32) for value in frame["action"]], dtype=np.float32)
        episodes.append(actions)
    return episodes


def audit(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bench_root = Path(args.bench_root).expanduser().resolve()
    hdf5_root = Path(args.hdf5_root).expanduser().resolve()
    specs = discover_step_specs(bench_root, hdf5_root, args.case_regex)
    if not specs:
        raise RuntimeError("No matching step specs found.")

    expected_by_case: dict[str, list[np.ndarray]] = {}
    for case_name in sorted({spec.case_name for spec in specs}):
        case_dir = bench_root / "Ideal" / case_name
        raw_actions = load_demo_actions(case_dir / "demo.hdf5")
        steps = parse_step_file(case_dir / "task_description.txt")
        expected_segments, _ = filtered_action_segments(raw_actions, steps)
        expected_by_case[case_name] = expected_segments

    lerobot_actions = read_lerobot_actions(Path(args.lerobot_dataset_root).expanduser().resolve()) if args.lerobot_dataset_root else []

    rows: list[dict[str, Any]] = []
    episode_idx = 0
    for spec in specs:
        expected_segments = expected_by_case[spec.case_name]
        expected = expected_segments[spec.step_index] if spec.step_index < len(expected_segments) else np.empty((0, 7), dtype=np.float32)
        per_step = load_per_step_actions(spec.hdf5_path) if spec.hdf5_path is not None else np.empty((0, 7), dtype=np.float32)
        lerobot = lerobot_actions[episode_idx] if episode_idx < len(lerobot_actions) else np.empty((0, 7), dtype=np.float32)
        episode_idx += 1

        expected_vs_hdf5 = max_abs_diff(expected, per_step)
        hdf5_vs_lerobot = max_abs_diff(per_step, lerobot) if len(lerobot_actions) else None
        hdf5_vs_lerobot_gripper_inverted = (
            max_abs_diff(per_step[:, -1] * -1.0, lerobot[:, -1])
            if len(lerobot_actions) and per_step.shape == lerobot.shape
            else None
        )

        rows.append(
            {
                "case_name": spec.case_name,
                "step_index": spec.step_index,
                "step_name": spec.step_name,
                "description": spec.description,
                "old_end_frame": spec.old_end_frame,
                "hdf5_path": str(spec.hdf5_path or ""),
                "expected_frames": len(expected),
                "per_step_hdf5_frames": len(per_step),
                "lerobot_frames": len(lerobot) if len(lerobot_actions) else "",
                "expected_vs_hdf5_max_abs_diff": expected_vs_hdf5 if expected_vs_hdf5 is not None else "",
                "hdf5_vs_lerobot_max_abs_diff": hdf5_vs_lerobot if hdf5_vs_lerobot is not None else "",
                "hdf5_vs_lerobot_gripper_inverted_max_abs_diff": (
                    hdf5_vs_lerobot_gripper_inverted if hdf5_vs_lerobot_gripper_inverted is not None else ""
                ),
                "expected_action_stats": json_cell(action_stats(expected)),
                "per_step_hdf5_action_stats": json_cell(action_stats(per_step)),
                "lerobot_action_stats": json_cell(action_stats(lerobot)) if len(lerobot_actions) else "",
                "frame_count_match": bool(len(expected) == len(per_step) and (not len(lerobot_actions) or len(per_step) == len(lerobot))),
                "hdf5_actions_match_expected": bool(expected_vs_hdf5 == 0.0),
                "lerobot_actions_match_hdf5": bool(hdf5_vs_lerobot == 0.0) if len(lerobot_actions) else "",
            }
        )

    summary = {
        "episodes": len(rows),
        "cases": len({row["case_name"] for row in rows}),
        "lerobot_episodes_found": len(lerobot_actions),
        "frame_mismatch_rows": sum(1 for row in rows if not row["frame_count_match"]),
        "hdf5_action_mismatch_rows": sum(1 for row in rows if not row["hdf5_actions_match_expected"]),
        "lerobot_action_mismatch_rows": sum(
            1 for row in rows if row["lerobot_actions_match_hdf5"] not in ("", True)
        ),
        "total_expected_frames": sum(int(row["expected_frames"]) for row in rows),
        "total_per_step_hdf5_frames": sum(int(row["per_step_hdf5_frames"]) for row in rows),
        "total_lerobot_frames": sum(int(row["lerobot_frames"]) for row in rows if row["lerobot_frames"] != ""),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench_root", required=True, help="RoboCerebraBench root.")
    parser.add_argument("--hdf5_root", required=True, help="Converted HDF5 root containing per_step/.")
    parser.add_argument("--lerobot_dataset_root", default=None, help="Optional final LeRobot dataset directory.")
    parser.add_argument("--case_regex", default=None, help="Optional regex over case names.")
    parser.add_argument("--output_csv", required=True, help="Output episode-level audit CSV.")
    args = parser.parse_args()

    rows, summary = audit(args)
    write_csv(Path(args.output_csv).expanduser(), rows)
    print("CONVERSION_AUDIT_COMPLETE")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  output_csv={Path(args.output_csv).expanduser()}")


if __name__ == "__main__":
    main()
