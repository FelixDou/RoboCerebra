#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch LeRobot fine-tuning for pi0 / pi0.5-style policies on RoboCerebra data.

This is a thin convenience wrapper around `lerobot-train` that standardizes the
local-dataset arguments and applies the documented pi0.5 normalization fallback
needed for datasets without quantile statistics.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PRETRAINED_PATHS = {
    "pi0": "lerobot/pi0_base",
    "pi05": "lerobot/pi05_base",
}


def canonicalize_model_family(model_family: str) -> str:
    normalized = str(model_family).strip().lower()
    if normalized in {"pi-zero", "pi_zero"}:
        return "pi0"
    if normalized in {"pi-05", "pi_05", "pi-0.5", "pi_0.5", "pi-zero-five", "pi_zero_five"}:
        return "pi05"
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch LeRobot fine-tuning for pi0 or pi0.5 on a local RoboCerebra LeRobotDataset."
    )
    parser.add_argument(
        "--model_family",
        required=True,
        choices=("pi0", "pi05", "pi-05", "pi-0.5", "pi_zero", "pi_zero_five", "pi-zero", "pi-zero-five"),
        help="Policy family to fine-tune.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        required=True,
        help="Local LeRobot dataset repo id / dataset name.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Root directory containing the local LeRobot dataset.",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        help="Base checkpoint or HF repo to fine-tune. Defaults to the corresponding LeRobot base policy.",
    )
    parser.add_argument(
        "--policy_repo_id",
        default=None,
        help="Optional policy repo id used by LeRobot logging/checkpoint metadata.",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        help="Training job name. Defaults to robocerebra_<model_family>_finetune.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Parent output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--policy_dtype",
        default="bfloat16",
        help="Policy dtype passed to LeRobot.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device passed to LeRobot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Training seed.",
    )
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable policy.compile_model.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable policy.gradient_checkpointing.",
    )
    parser.add_argument(
        "--freeze_vision_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable policy.freeze_vision_encoder.",
    )
    parser.add_argument(
        "--train_expert_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable policy.train_expert_only.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable wandb logging.",
    )
    parser.add_argument(
        "--normalization_mapping",
        default=None,
        help="Optional explicit normalization mapping override passed to LeRobot.",
    )
    parser.add_argument(
        "--disable_default_pi05_norm_fallback",
        action="store_true",
        help="Do not inject the documented pi0.5 mean/std normalization fallback.",
    )
    parser.add_argument(
        "--train_entrypoint",
        default=None,
        help="Optional explicit training executable. Defaults to `lerobot-train`, then falls back to `python -m lerobot.scripts.lerobot_train`.",
    )
    parser.add_argument(
        "--extra_arg",
        action="append",
        default=[],
        help="Extra raw argument appended verbatim to the underlying `lerobot-train` command. Repeatable.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args()


def append_override(command: list[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        normalized_value = "true" if value else "false"
    else:
        normalized_value = str(value)
    command.append(f"--{key}={normalized_value}")


def resolve_train_command(explicit_entrypoint: str | None) -> list[str]:
    if explicit_entrypoint:
        return [explicit_entrypoint]

    lerobot_train = shutil.which("lerobot-train")
    if lerobot_train:
        return [lerobot_train]

    python_executable = shutil.which("python")
    if python_executable:
        return [python_executable, "-m", "lerobot.scripts.lerobot_train"]

    raise FileNotFoundError(
        "Could not find `lerobot-train` or a usable `python` executable. "
        "Install LeRobot first, or pass --train_entrypoint explicitly."
    )


def default_policy_repo_id(model_family: str, job_name: str) -> str:
    del model_family
    repo_suffix = job_name.removeprefix("robocerebra_")
    return f"robocerebra/{repo_suffix}"


def build_command(args: argparse.Namespace) -> list[str]:
    model_family = canonicalize_model_family(args.model_family)
    if model_family not in DEFAULT_PRETRAINED_PATHS:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    job_name = args.job_name or f"robocerebra_{model_family}_finetune"
    output_dir = str(Path(args.output_dir).expanduser().resolve() / job_name)
    pretrained_path = args.pretrained_path or DEFAULT_PRETRAINED_PATHS[model_family]
    policy_repo_id = args.policy_repo_id or default_policy_repo_id(model_family, job_name)

    command = resolve_train_command(args.train_entrypoint)
    append_override(command, "dataset.repo_id", args.dataset_repo_id)
    append_override(command, "dataset.root", str(Path(args.dataset_root).expanduser().resolve()))
    append_override(command, "policy.type", model_family)
    append_override(command, "policy.repo_id", policy_repo_id)
    append_override(command, "policy.pretrained_path", pretrained_path)
    append_override(command, "output_dir", output_dir)
    append_override(command, "job_name", job_name)
    append_override(command, "steps", args.steps)
    append_override(command, "batch_size", args.batch_size)
    append_override(command, "seed", args.seed)
    append_override(command, "policy.device", args.device)
    append_override(command, "policy.dtype", args.policy_dtype)
    append_override(command, "policy.compile_model", args.compile_model)
    append_override(command, "policy.gradient_checkpointing", args.gradient_checkpointing)
    append_override(command, "policy.freeze_vision_encoder", args.freeze_vision_encoder)
    append_override(command, "policy.train_expert_only", args.train_expert_only)
    append_override(command, "wandb.enable", args.wandb)

    normalization_mapping = args.normalization_mapping
    if (
        model_family == "pi05"
        and normalization_mapping is None
        and not args.disable_default_pi05_norm_fallback
    ):
        normalization_mapping = '{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
    append_override(command, "policy.normalization_mapping", normalization_mapping)

    for extra_arg in args.extra_arg:
        command.append(extra_arg)

    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print("Resolved LeRobot training command:")
    print(shlex.join(command))

    if args.dry_run:
        return

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
