#!/usr/bin/env python3
"""Compare benchmark GT actions against PI-family policy actions open-loop.

This diagnostic starts both rollouts from the same recorded benchmark state for
one task segment. It then executes either the recorded demo actions or the
policy's predicted actions and writes videos plus a per-step action trace.

Use this when closed-loop evaluation looks random: it separates action
representation / normalization errors from benchmark reward-monitor noise.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import GenerateConfig
from policy_adapter import initialize_policy, predict_policy_actions, reset_policy_state
from replay_gt_actions import load_demo, parse_step_intervals, set_env_state
from task_runner import setup_task_environment
from utils import prepare_observation, process_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocerebra_root", required=True, help="Path to RoboCerebraBench.")
    parser.add_argument("--checkpoint", required=True, help="PI0/PI05 checkpoint or HF repo id.")
    parser.add_argument("--task_type", default="Ideal")
    parser.add_argument("--case_name", default="case5")
    parser.add_argument("--step_index", type=int, default=0)
    parser.add_argument("--model_family", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=None, help="Optional cap within the selected segment.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_cfg(args: argparse.Namespace) -> GenerateConfig:
    cfg = GenerateConfig()
    cfg.model_family = args.model_family
    cfg.pretrained_checkpoint = args.checkpoint
    cfg.robocerebra_root = args.robocerebra_root
    cfg.init_files_root = str(Path(args.robocerebra_root) / "init_files")
    cfg.task_types = [args.task_type]
    cfg.seed = args.seed
    cfg.use_init_files = True
    cfg.resume = False
    return cfg


def capture_frame(env: Any) -> np.ndarray:
    obs = env._get_observations()
    _, frame = prepare_observation(obs, resize_size=None)
    return np.asarray(frame, dtype=np.uint8)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.stack(frames), fps=fps)
        return
    except Exception:
        pass

    import imageio

    imageio.mimsave(path, frames, fps=fps)


def rollout_gt(env: Any, start_state: np.ndarray, gt_actions: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    env.reset()
    set_env_state(env, start_state)

    frames: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    for action in gt_actions:
        frames.append(capture_frame(env))
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        env.step(action.tolist())
        executed.append(action)
    frames.append(capture_frame(env))
    return frames, np.stack(executed)


def rollout_policy(
    env: Any,
    cfg: GenerateConfig,
    policy_runtime: Any,
    start_state: np.ndarray,
    desc: str,
    steps: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    env.reset()
    set_env_state(env, start_state)
    reset_policy_state(policy_runtime)

    frames: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    for _ in range(steps):
        obs = env._get_observations()
        observation, frame = prepare_observation(obs, resize_size=policy_runtime.resize_size)
        frames.append(np.asarray(frame, dtype=np.uint8))
        pred_actions = predict_policy_actions(cfg, policy_runtime, observation, desc)
        if not pred_actions:
            raise RuntimeError("Policy returned no actions.")
        action = process_action(pred_actions[0], cfg.model_family)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        env.step(action.tolist())
        executed.append(action)
    frames.append(capture_frame(env))
    return frames, np.stack(executed)


def write_trace_csv(path: Path, gt_actions: np.ndarray, policy_actions: np.ndarray, start_frame: int) -> dict[str, Any]:
    if gt_actions.shape != policy_actions.shape:
        raise ValueError(f"Action shape mismatch: gt={gt_actions.shape}, policy={policy_actions.shape}")

    diff = policy_actions - gt_actions
    denom = np.linalg.norm(gt_actions, axis=1) * np.linalg.norm(policy_actions, axis=1)
    cosine = np.divide(
        np.sum(gt_actions * policy_actions, axis=1),
        denom,
        out=np.zeros(len(gt_actions), dtype=np.float32),
        where=denom > 1e-8,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    dim_count = gt_actions.shape[1]
    header = (
        ["t", "global_frame"]
        + [f"gt_action_{idx}" for idx in range(dim_count)]
        + [f"policy_action_{idx}" for idx in range(dim_count)]
        + [f"diff_{idx}" for idx in range(dim_count)]
        + ["gt_norm", "policy_norm", "diff_norm", "cosine"]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for t in range(len(gt_actions)):
            writer.writerow(
                [t, start_frame + t]
                + gt_actions[t].astype(float).tolist()
                + policy_actions[t].astype(float).tolist()
                + diff[t].astype(float).tolist()
                + [
                    float(np.linalg.norm(gt_actions[t])),
                    float(np.linalg.norm(policy_actions[t])),
                    float(np.linalg.norm(diff[t])),
                    float(cosine[t]),
                ]
            )

    return {
        "frames": int(len(gt_actions)),
        "action_mae": float(np.mean(np.abs(diff))),
        "action_max_abs": float(np.max(np.abs(diff))),
        "per_dim_mae": np.mean(np.abs(diff), axis=0).astype(float).tolist(),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "gt_norm_mean": float(np.mean(np.linalg.norm(gt_actions, axis=1))),
        "policy_norm_mean": float(np.mean(np.linalg.norm(policy_actions, axis=1))),
        "gripper_mae": float(np.mean(np.abs(diff[:, -1]))),
    }


def make_side_by_side(gt_frames: list[np.ndarray], policy_frames: list[np.ndarray]) -> list[np.ndarray]:
    frame_count = min(len(gt_frames), len(policy_frames))
    side_by_side: list[np.ndarray] = []
    for idx in range(frame_count):
        left = gt_frames[idx]
        right = policy_frames[idx]
        if left.shape != right.shape:
            raise ValueError(f"Frame shape mismatch at {idx}: {left.shape} vs {right.shape}")
        separator = np.zeros((left.shape[0], 4, left.shape[2]), dtype=np.uint8)
        side_by_side.append(np.concatenate([left, separator, right], axis=1))
    return side_by_side


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)
    root = Path(args.robocerebra_root).expanduser().resolve()
    task_dir = root / args.task_type / args.case_name
    output_dir = Path(args.output_dir).expanduser().resolve()

    intervals = parse_step_intervals(task_dir / "task_description.txt")
    if args.step_index < 0 or args.step_index >= len(intervals):
        raise ValueError(f"--step_index {args.step_index} out of range for {len(intervals)} intervals")

    desc, start, end = intervals[args.step_index]
    states, actions = load_demo(task_dir / "demo.hdf5")
    start = max(0, min(start, len(states) - 1))
    end = max(start + 1, min(end, len(actions)))
    if args.max_steps is not None:
        end = min(end, start + max(1, args.max_steps))

    env, _, error = setup_task_environment(task_dir)
    if error or env is None:
        raise RuntimeError(error or f"Failed to create environment for {task_dir}")

    try:
        policy_runtime = initialize_policy(cfg)

        gt_frames, gt_executed = rollout_gt(env, states[start], actions[start:end])
        policy_frames, policy_executed = rollout_policy(
            env,
            cfg,
            policy_runtime,
            states[start],
            desc,
            steps=end - start,
        )

        gt_video = output_dir / f"{args.case_name}_step{args.step_index}_gt.mp4"
        policy_video = output_dir / f"{args.case_name}_step{args.step_index}_{args.model_family}.mp4"
        side_by_side_video = output_dir / f"{args.case_name}_step{args.step_index}_gt_vs_{args.model_family}.mp4"
        trace_csv = output_dir / f"{args.case_name}_step{args.step_index}_action_trace.csv"
        summary_json = output_dir / f"{args.case_name}_step{args.step_index}_summary.json"

        write_video(gt_video, gt_frames, fps=args.fps)
        write_video(policy_video, policy_frames, fps=args.fps)
        write_video(side_by_side_video, make_side_by_side(gt_frames, policy_frames), fps=args.fps)
        summary = write_trace_csv(trace_csv, gt_executed, policy_executed, start_frame=start)
        summary.update(
            {
                "task_dir": str(task_dir),
                "checkpoint": str(args.checkpoint),
                "model_family": args.model_family,
                "step_index": int(args.step_index),
                "description": desc,
                "frame_start": int(start),
                "frame_end": int(end),
                "gt_video": str(gt_video),
                "policy_video": str(policy_video),
                "side_by_side_video": str(side_by_side_video),
                "trace_csv": str(trace_csv),
            }
        )
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        print("OPEN_LOOP_ACTION_COMPARISON_COMPLETE")
        for key in (
            "frames",
            "action_mae",
            "action_max_abs",
            "gripper_mae",
            "cosine_mean",
            "cosine_min",
            "gt_norm_mean",
            "policy_norm_mean",
            "per_dim_mae",
            "gt_video",
            "policy_video",
            "side_by_side_video",
            "trace_csv",
            "summary_json",
        ):
            value = str(summary_json) if key == "summary_json" else summary[key]
            print(f"  {key}={value}")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
