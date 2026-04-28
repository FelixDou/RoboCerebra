#!/usr/bin/env python3
"""Replay RoboCerebra benchmark GT actions in the evaluation environment.

This diagnostic separates policy-learning failures from environment / action /
init-state mismatches. It runs the `demo.hdf5` actions directly through the same
LIBERO environment and goal checker used by evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from config import GenerateConfig
from resume import create_step_based_resume_handler, simulate_resume_completion
from task_runner import setup_task_environment
from utils import get_libero_dummy_action, load_actions_with_steps, load_init_state


def parse_step_intervals(task_description_path: Path) -> list[tuple[str, int, int]]:
    intervals: list[tuple[str, int, int]] = []
    lines = [line.strip() for line in task_description_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bracket_re = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("Step"):
            idx += 1
            continue
        desc = line.split(":", 1)[1].strip()
        if idx + 1 >= len(lines):
            raise ValueError(f"Missing frame interval after step: {desc}")
        match = bracket_re.match(lines[idx + 1])
        if not match:
            raise ValueError(f"Expected [start, end] after step `{desc}`, found: {lines[idx + 1]}")
        intervals.append((desc, int(match.group(1)), int(match.group(2))))
        idx += 2

    if not intervals:
        raise ValueError(f"No step intervals found in {task_description_path}")
    return intervals


def load_demo(demo_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(demo_path, "r") as handle:
        demo = handle["data"]["demo_1"]
        states = np.asarray(demo["states"][()])
        actions = np.asarray(demo["actions"][()], dtype=np.float32)
    if len(states) != len(actions):
        raise ValueError(f"states/actions length mismatch in {demo_path}: {len(states)} vs {len(actions)}")
    return states, actions


def set_env_state(env, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)


def step_actions(env, actions: Iterable[np.ndarray], goal=None, initial_completed: int | None = None) -> int:
    """Replay actions and optionally poll the benchmark success monitor after every step."""
    total_completed_prev = initial_completed
    positive_delta = 0
    for action in actions:
        env.step(np.asarray(action, dtype=np.float32).tolist())
        if goal is None:
            continue
        _, total_completed_now, _ = env._check_success(goal)
        total_completed_now = int(total_completed_now)
        if total_completed_prev is not None:
            positive_delta += max(0, total_completed_now - total_completed_prev)
        total_completed_prev = total_completed_now
    return positive_delta


def completed_count(env, goal) -> tuple[int, dict]:
    details, total, _ = env._check_success(goal)
    return int(total), details


def object_regions(env, obj: str) -> list[str]:
    regions = [region for region in env.regions if env._eval_predicate(["in", obj, region])]
    return regions or ["None"]


def evaluate_raw_predicate(env, predicate: list[str]) -> bool:
    if len(predicate) == 2:
        return bool(env._eval_predicate(predicate))
    if len(predicate) == 3:
        return bool(env._eval_predicate(predicate))
    raise ValueError(f"Unsupported predicate format: {predicate}")


def run_continuous_replay(env, goal, states: np.ndarray, actions: np.ndarray, initial_state: np.ndarray | None) -> None:
    env.reset()
    set_env_state(env, states[0])
    env.skip_pick_quat_once = True
    step_actions(env, [get_libero_dummy_action("pi0")] * 15)
    before, before_details = completed_count(env, goal)
    positive_delta = step_actions(env, actions, goal=goal, initial_completed=before)
    after, after_details = completed_count(env, goal)
    print("CONTINUOUS_REPLAY_FROM_DEMO_STATE0")
    print(f"  start_completed={before} details={json.dumps(before_details, sort_keys=True)}")
    print(f"  polled_positive_delta={positive_delta}")
    print(f"  final_completed={after} details={json.dumps(after_details, sort_keys=True)}")

    if initial_state is not None:
        env.reset()
        set_env_state(env, initial_state)
        step_actions(env, [get_libero_dummy_action("pi0")] * 15)
        before, before_details = completed_count(env, goal)
        positive_delta = step_actions(env, actions, goal=goal, initial_completed=before)
        after, after_details = completed_count(env, goal)
        print("CONTINUOUS_REPLAY_FROM_INIT_FILE")
        print(f"  start_completed={before} details={json.dumps(before_details, sort_keys=True)}")
        print(f"  polled_positive_delta={positive_delta}")
        print(f"  final_completed={after} details={json.dumps(after_details, sort_keys=True)}")


def run_segmented_replay(
    env,
    goal,
    states: np.ndarray,
    actions: np.ndarray,
    intervals: list[tuple[str, int, int]],
    resume_handler,
) -> None:
    print("SEGMENTED_REPLAY_WITH_EVAL_RESUME")
    total_after = 0
    for step_idx, (desc, start, end) in enumerate(intervals):
        env.reset()
        start = max(0, min(start, len(states) - 1))
        end = max(start + 1, min(end, len(actions)))
        set_env_state(env, states[start])
        env.skip_pick_quat_once = True
        # The evaluator calls _check_success at the previous segment boundary
        # before applying resume bookkeeping. Do the same after resetting a
        # standalone segment so env._state_progress exists.
        completed_count(env, goal)
        resume_count, completed_by_resume = simulate_resume_completion(env, goal, resume_handler, step_idx)
        if step_idx == 0:
            step_actions(env, [get_libero_dummy_action("pi0")] * 15)
        before, before_details = completed_count(env, goal)
        polled_positive_delta = step_actions(env, actions[start:end], goal=goal, initial_completed=before)
        after, after_details = completed_count(env, goal)
        total_after += polled_positive_delta
        print(
            f"  step={step_idx} frames=[{start},{end}) resume_count={resume_count} "
            f"completed_by_resume={completed_by_resume} before={before} after={after} "
            f"final_delta={after - before} polled_positive_delta={polled_positive_delta} desc={desc!r}"
        )
        print(f"    before_details={json.dumps(before_details, sort_keys=True)}")
        print(f"    after_details={json.dumps(after_details, sort_keys=True)}")
    print(f"  summed_polled_positive_segment_deltas={total_after}")


def scan_demo_states(env, goal, states: np.ndarray, intervals: list[tuple[str, int, int]]) -> None:
    """Scan recorded states directly through the goal monitor.

    This bypasses action execution. If this does not reach all goals, the issue is
    likely goal annotations / monitor semantics / demo-state compatibility rather
    than policy training.
    """
    env.reset()
    max_completed = 0
    first_reach: dict[int, int] = {}
    last_details = None

    for frame_idx, state in enumerate(states):
        set_env_state(env, state)
        _, total_completed_now, _ = env._check_success(goal)
        total_completed_now = int(total_completed_now)
        last_details = env._check_success(goal)[0]
        if total_completed_now > max_completed:
            for count in range(max_completed + 1, total_completed_now + 1):
                first_reach[count] = frame_idx
            max_completed = total_completed_now

    print("DEMO_STATE_SCAN")
    print(f"  max_completed={max_completed}")
    print(f"  first_reach={json.dumps(first_reach, sort_keys=True)}")
    print(f"  final_details={json.dumps(last_details, sort_keys=True)}")

    for step_idx, (desc, start, end) in enumerate(intervals):
        env.reset()
        step_max = 0
        step_first_reach: dict[int, int] = {}
        for frame_idx in range(max(0, start), min(end, len(states))):
            set_env_state(env, states[frame_idx])
            _, total_completed_now, _ = env._check_success(goal)
            total_completed_now = int(total_completed_now)
            if total_completed_now > step_max:
                for count in range(step_max + 1, total_completed_now + 1):
                    step_first_reach[count] = frame_idx
                step_max = total_completed_now
        print(f"  segment={step_idx} frames=[{start},{end}) max_completed={step_max} first_reach={json.dumps(step_first_reach, sort_keys=True)} desc={desc!r}")


def scan_failed_predicates(env, goal, states: np.ndarray, intervals: list[tuple[str, int, int]]) -> None:
    """Report raw goal predicates that never become true in recorded states."""
    print("FAILED_PREDICATE_SCAN")
    for obj, predicates in goal.items():
        observed_regions: set[str] = set()
        predicate_hits: list[dict[str, object]] = []

        for pred_idx, predicate in enumerate(predicates):
            first_true = None
            segment_idx = None
            target = predicate[2] if len(predicate) == 3 else None

            env.reset()
            for frame_idx, state in enumerate(states):
                set_env_state(env, state)
                observed_regions.update(object_regions(env, obj))
                if evaluate_raw_predicate(env, predicate):
                    first_true = frame_idx
                    for idx, (_, start, end) in enumerate(intervals):
                        if start <= frame_idx < end:
                            segment_idx = idx
                            break
                    break

            set_env_state(env, states[-1])
            final_regions = object_regions(env, obj)
            predicate_hits.append(
                {
                    "predicate_index": pred_idx,
                    "predicate": predicate,
                    "target": target,
                    "first_true_frame": first_true,
                    "first_true_segment": segment_idx,
                    "final_regions": final_regions,
                }
            )

        missed = [entry for entry in predicate_hits if entry["first_true_frame"] is None]
        print(f"  object={obj} observed_regions={sorted(observed_regions)}")
        for entry in predicate_hits:
            status = "HIT" if entry["first_true_frame"] is not None else "MISS"
            print(
                "    "
                f"{status} idx={entry['predicate_index']} predicate={entry['predicate']} "
                f"target={entry['target']} first_true_frame={entry['first_true_frame']} "
                f"first_true_segment={entry['first_true_segment']} final_regions={entry['final_regions']}"
            )
        if missed:
            print(f"    missed_count={len(missed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocerebra_root", required=True, help="RoboCerebraBench root.")
    parser.add_argument("--init_files_root", default=None, help="Optional init_files root.")
    parser.add_argument("--task_type", default="Ideal", help="Benchmark task type directory.")
    parser.add_argument("--case_name", default="case1", help="Case directory name.")
    args = parser.parse_args()

    root = Path(args.robocerebra_root).expanduser().resolve()
    task_dir = root / args.task_type / args.case_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Missing task directory: {task_dir}")

    env, _, error = setup_task_environment(task_dir)
    if error:
        raise RuntimeError(error)
    if env is None:
        raise RuntimeError(f"Failed to initialize environment for {task_dir}")

    states, actions = load_demo(task_dir / "demo.hdf5")
    intervals = parse_step_intervals(task_dir / "task_description.txt")
    goal, goal_steps = load_actions_with_steps(str(task_dir / "goal.json"))
    resume_handler = create_step_based_resume_handler(goal, goal_steps)

    cfg = GenerateConfig(
        robocerebra_root=str(root),
        init_files_root=str(args.init_files_root or (root / "init_files")),
        task_types=[args.task_type],
    )
    initial_state = load_init_state(cfg, args.task_type, args.case_name)

    print(f"TASK_DIR={task_dir}")
    print(f"DEMO_STATES={len(states)} DEMO_ACTIONS={len(actions)}")
    print("INTERVALS=" + ", ".join(f"{idx}:{start}-{end}" for idx, (_, start, end) in enumerate(intervals)))
    print(f"GOAL_STEPS={json.dumps(goal_steps, sort_keys=True)}")
    print(f"RESUME_HANDLER={json.dumps(resume_handler, sort_keys=True)}")
    print(f"INIT_STATE={'yes' if initial_state is not None else 'no'}")

    run_continuous_replay(env, goal, states, actions, initial_state)
    run_segmented_replay(env, goal, states, actions, intervals, resume_handler)
    scan_demo_states(env, goal, states, intervals)
    scan_failed_predicates(env, goal, states, intervals)


if __name__ == "__main__":
    main()
