#!/usr/bin/env python3
"""Audit RoboCerebra GT demos against their goal predicates.

This script is intended to answer a narrow question before policy debugging:
do recorded `demo.hdf5` states ever satisfy the predicates listed in
`goal.json` under the same LIBERO/RoboCerebra success code used by evaluation?

It writes a predicate-level CSV so invalid cases can be filtered or corrected
before training / evaluation numbers are interpreted.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from replay_gt_actions import (
    evaluate_raw_predicate,
    load_demo,
    object_regions,
    parse_step_intervals,
    set_env_state,
)
from task_runner import setup_task_environment
from utils import load_actions_with_steps


SOURCE_REGION_HINTS = (
    "coffee_table",
    "dining_set",
    "wooden_cabinet",
    "rack",
)
SPATIAL_TARGET_VERBS = {"in", "on"}


@dataclass(frozen=True)
class DemoScan:
    max_completed: int
    first_reach: dict[int, int]
    final_details: dict[str, Any] | None


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def segment_for_frame(frame_idx: int | None, intervals: list[tuple[str, int, int]]) -> int | None:
    if frame_idx is None:
        return None
    for idx, (_, start, end) in enumerate(intervals):
        if start <= frame_idx < end:
            return idx
    return None


def scan_demo_progress(env, goal: dict[str, list[list[str]]], states: np.ndarray) -> DemoScan:
    env.reset()
    max_completed = 0
    first_reach: dict[int, int] = {}
    final_details = None

    for frame_idx, state in enumerate(states):
        set_env_state(env, state)
        final_details, total_completed_now, _ = env._check_success(goal)
        total_completed_now = int(total_completed_now)
        if total_completed_now > max_completed:
            for count in range(max_completed + 1, total_completed_now + 1):
                first_reach[count] = frame_idx
            max_completed = total_completed_now

    return DemoScan(max_completed=max_completed, first_reach=first_reach, final_details=final_details)


def scan_region_counts(env, obj: str, states: np.ndarray) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    final_regions: list[str] = ["None"]
    env.reset()
    for state in states:
        set_env_state(env, state)
        regions = object_regions(env, obj)
        counts.update(regions)
        final_regions = regions
    return counts, final_regions


def scan_predicate_first_true(
    env,
    predicate: list[str],
    states: np.ndarray,
) -> int | None:
    env.reset()
    for frame_idx, state in enumerate(states):
        set_env_state(env, state)
        if evaluate_raw_predicate(env, predicate):
            return frame_idx
    return None


def infer_suggested_target(
    predicate: list[str],
    target: str | None,
    region_counts: Counter[str],
    final_regions: list[str],
) -> str:
    if len(predicate) != 3 or target is None or predicate[0] not in SPATIAL_TARGET_VERBS:
        return ""
    if target in region_counts:
        return ""

    candidates = [region for region in final_regions if region != "None" and region != target]
    if not candidates:
        candidates = [
            region
            for region, _ in region_counts.most_common()
            if region != "None" and region != target
        ]
    if not candidates:
        return ""

    # Prefer a concrete destination that was observed in the demo, but avoid
    # presenting the initial source as a confident fix when nothing moved.
    destination_candidates = [
        region
        for region in candidates
        if not any(hint in region for hint in SOURCE_REGION_HINTS)
    ]
    if not destination_candidates:
        return ""
    return destination_candidates[0]


def predicate_issue(
    status: str,
    target: str | None,
    suggested_target: str,
    first_true_segment: int | None,
    expected_step: int | None,
) -> str:
    issues: list[str] = []
    if status == "MISS":
        issues.append("predicate_never_true")
    if target and suggested_target:
        issues.append("target_region_mismatch")
    if (
        first_true_segment is not None
        and expected_step is not None
        and first_true_segment != expected_step
    ):
        issues.append("step_boundary_or_order_mismatch")
    return ";".join(issues)


def audit_case(root: Path, task_type: str, case_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env, _, error = setup_task_environment(case_dir)
    if error:
        raise RuntimeError(error)
    if env is None:
        raise RuntimeError(f"Failed to initialize environment for {case_dir}")

    states, actions = load_demo(case_dir / "demo.hdf5")
    intervals = parse_step_intervals(case_dir / "task_description.txt")
    goal, goal_steps = load_actions_with_steps(str(case_dir / "goal.json"))
    demo_scan = scan_demo_progress(env, goal, states)
    total_goal_predicates = sum(len(predicates) for predicates in goal.values())

    rows: list[dict[str, Any]] = []
    hit_count = 0
    miss_count = 0

    for obj, predicates in goal.items():
        region_counts, final_regions = scan_region_counts(env, obj, states)
        observed_regions = sorted(region_counts)
        for pred_idx, predicate in enumerate(predicates):
            target = predicate[2] if len(predicate) == 3 else None
            expected_step = goal_steps.get(obj, [None] * len(predicates))[pred_idx]
            step_desc = intervals[expected_step][0] if expected_step is not None and 0 <= expected_step < len(intervals) else ""
            first_true_frame = scan_predicate_first_true(env, predicate, states)
            first_true_segment = segment_for_frame(first_true_frame, intervals)
            status = "HIT" if first_true_frame is not None else "MISS"
            suggested_target = infer_suggested_target(predicate, target, region_counts, final_regions)

            if status == "HIT":
                hit_count += 1
            else:
                miss_count += 1

            rows.append(
                {
                    "task_type": task_type,
                    "case_name": case_dir.name,
                    "case_dir": str(case_dir),
                    "object": obj,
                    "predicate_index": pred_idx,
                    "predicate": json_cell(predicate),
                    "verb": predicate[0] if predicate else "",
                    "target": target or "",
                    "status": status,
                    "issue": predicate_issue(status, target, suggested_target, first_true_segment, expected_step),
                    "expected_step": expected_step if expected_step is not None else "",
                    "expected_step_description": step_desc,
                    "first_true_frame": first_true_frame if first_true_frame is not None else "",
                    "first_true_segment": first_true_segment if first_true_segment is not None else "",
                    "suggested_target_region": suggested_target,
                    "observed_regions": json_cell(observed_regions),
                    "region_counts": json_cell(dict(region_counts)),
                    "final_regions": json_cell(final_regions),
                    "demo_states": len(states),
                    "demo_actions": len(actions),
                    "total_goal_predicates": total_goal_predicates,
                    "demo_state_scan_max_completed": demo_scan.max_completed,
                    "demo_state_scan_first_reach": json_cell(demo_scan.first_reach),
                    "demo_state_scan_final_details": json_cell(demo_scan.final_details),
                }
            )

    summary = {
        "task_type": task_type,
        "case_name": case_dir.name,
        "case_dir": str(case_dir),
        "demo_states": len(states),
        "demo_actions": len(actions),
        "total_goal_predicates": total_goal_predicates,
        "raw_predicate_hits": hit_count,
        "raw_predicate_misses": miss_count,
        "demo_state_scan_max_completed": demo_scan.max_completed,
        "fully_raw_satisfied": miss_count == 0,
        "fully_sequentially_satisfied": demo_scan.max_completed >= total_goal_predicates,
    }
    return rows, summary


def discover_cases(root: Path, task_types: list[str], case_regex: str | None, max_cases: int | None) -> list[tuple[str, Path]]:
    import re

    pattern = re.compile(case_regex) if case_regex else None
    cases: list[tuple[str, Path]] = []
    for task_type in task_types:
        task_root = root / task_type
        if not task_root.is_dir():
            raise FileNotFoundError(f"Missing task type directory: {task_root}")
        for case_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            if pattern and not pattern.search(case_dir.name):
                continue
            if not (case_dir / "demo.hdf5").is_file():
                continue
            if not (case_dir / "goal.json").is_file():
                continue
            if not (case_dir / "task_description.txt").is_file():
                continue
            cases.append((task_type, case_dir))
            if max_cases is not None and len(cases) >= max_cases:
                return cases
    return cases


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocerebra_root", required=True, help="RoboCerebraBench root.")
    parser.add_argument(
        "--task_type",
        action="append",
        default=None,
        help="Benchmark task type to audit. Repeat for multiple types. Default: Ideal.",
    )
    parser.add_argument("--case_regex", default=None, help="Optional regex over case directory names.")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--output_csv", required=True, help="Predicate-level audit CSV path.")
    parser.add_argument("--summary_csv", default=None, help="Optional case-level summary CSV path.")
    args = parser.parse_args()

    root = Path(args.robocerebra_root).expanduser().resolve()
    task_types = args.task_type or ["Ideal"]
    cases = discover_cases(root, task_types, args.case_regex, args.max_cases)
    if not cases:
        raise RuntimeError(f"No matching cases found under {root} for task types {task_types}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for idx, (task_type, case_dir) in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Auditing {task_type}/{case_dir.name}")
        rows, summary = audit_case(root, task_type, case_dir)
        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(Path(args.output_csv).expanduser(), all_rows)
    if args.summary_csv:
        write_csv(Path(args.summary_csv).expanduser(), summaries)

    total_predicates = len(all_rows)
    total_misses = sum(1 for row in all_rows if row["status"] == "MISS")
    suspicious = sum(1 for row in all_rows if row["issue"])
    invalid_sequential = sum(1 for row in summaries if not row["fully_sequentially_satisfied"])
    print("AUDIT_COMPLETE")
    print(f"  cases={len(summaries)} predicates={total_predicates} misses={total_misses} suspicious_rows={suspicious}")
    print(f"  cases_not_fully_sequentially_satisfied={invalid_sequential}")
    print(f"  output_csv={Path(args.output_csv).expanduser()}")
    if args.summary_csv:
        print(f"  summary_csv={Path(args.summary_csv).expanduser()}")


if __name__ == "__main__":
    main()
