#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch OpenVLA-OFT fine-tuning on a RoboCerebra RLDS / TFDS dataset.

This wrapper mirrors the role of ``training/finetune_lerobot_policy.py`` for PI
models, but targets OpenVLA-OFT's upstream ``vla-scripts/finetune.py`` entry
point instead of LeRobot. It intentionally stays thin: dataset loading, model
definition, LoRA, action heads, checkpointing, and validation remain owned by
the OpenVLA-OFT repository.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


DEFAULT_VLA_PATH = "openvla/openvla-7b"
DEFAULT_JOB_NAME = "robocerebra_openvla_oft_finetune"


def parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch OpenVLA-OFT LoRA fine-tuning on a local RoboCerebra RLDS dataset."
    )
    parser.add_argument(
        "--openvla_oft_root",
        default=os.environ.get("OPENVLA_OFT_ROOT"),
        help=(
            "Path to a local moojink/openvla-oft checkout. Defaults to OPENVLA_OFT_ROOT, "
            "then ../openvla-oft if present."
        ),
    )
    parser.add_argument(
        "--finetune_script",
        default=None,
        help="Optional explicit path to OpenVLA-OFT vla-scripts/finetune.py.",
    )
    parser.add_argument(
        "--rlds_dir",
        default=None,
        help=(
            "Optional path to one TFDS dataset directory. If provided, --data_root_dir and "
            "--dataset_name are inferred. Pass either the dataset root or its version directory."
        ),
    )
    parser.add_argument(
        "--data_root_dir",
        default=None,
        help="Directory containing the RLDS / TFDS dataset directory expected by OpenVLA-OFT.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="TFDS dataset name under --data_root_dir, e.g. homerobo_dataset.",
    )
    parser.add_argument(
        "--vla_path",
        default=DEFAULT_VLA_PATH,
        help="Base OpenVLA checkpoint or local checkpoint path.",
    )
    parser.add_argument(
        "--job_name",
        default=DEFAULT_JOB_NAME,
        help="Stable run label used as OpenVLA-OFT run_id_note unless --run_id_note is set.",
    )
    parser.add_argument(
        "--run_id_note",
        default=None,
        help="OpenVLA-OFT run_id_note. Defaults to --job_name.",
    )
    parser.add_argument(
        "--run_id_override",
        default=None,
        help="Optional exact OpenVLA-OFT run id override.",
    )
    parser.add_argument(
        "--run_root_dir",
        default="./outputs/openvla_oft",
        help="Directory for OpenVLA-OFT logs and checkpoints.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of GPUs passed to torchrun --nproc-per-node.",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes for torchrun.")
    parser.add_argument("--torchrun", default=None, help="Optional explicit torchrun executable.")
    parser.add_argument("--standalone", type=parse_bool, default=True, help="Pass --standalone to torchrun.")
    parser.add_argument("--master_port", default=None, help="Optional MASTER_PORT for torchrun.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--max_steps", type=int, default=50_000, help="Number of gradient steps.")
    parser.add_argument("--save_freq", type=int, default=5_000, help="Checkpoint save frequency.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument(
        "--num_steps_before_decay",
        type=int,
        default=100_000,
        help="OpenVLA-OFT MultiStepLR decay milestone.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps passed to OpenVLA-OFT.",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=100_000,
        help="RLDS shuffle buffer size. Reduce this if memory or startup time is high.",
    )
    parser.add_argument("--use_l1_regression", type=parse_bool, default=True)
    parser.add_argument("--use_diffusion", type=parse_bool, default=False)
    parser.add_argument("--use_film", type=parse_bool, default=False)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--use_proprio", type=parse_bool, default=True)
    parser.add_argument("--image_aug", type=parse_bool, default=True)
    parser.add_argument("--use_lora", type=parse_bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--merge_lora_during_training", type=parse_bool, default=True)
    parser.add_argument("--save_latest_checkpoint_only", type=parse_bool, default=False)
    parser.add_argument("--resume", type=parse_bool, default=False)
    parser.add_argument("--resume_step", type=int, default=None)
    parser.add_argument("--use_val_set", type=parse_bool, default=False)
    parser.add_argument("--val_freq", type=int, default=10_000)
    parser.add_argument("--val_time_limit", type=int, default=180)
    parser.add_argument("--wandb_entity", default="felixdoublet")
    parser.add_argument("--wandb_project", default="robocerebra-openvla")
    parser.add_argument("--wandb_log_freq", type=int, default=10)
    parser.add_argument(
        "--extra_arg",
        action="append",
        default=[],
        help="Extra raw OpenVLA-OFT argument appended as-is. Repeatable.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def bool_string(value: bool) -> str:
    return "True" if value else "False"


def append_arg(command: list[str], name: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = bool_string(value)
    command.extend([f"--{name}", str(value)])


def resolve_openvla_oft_root(raw_root: str | None, explicit_script: str | None) -> Path:
    if raw_root:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"OpenVLA-OFT root does not exist: {root}")
        return root

    if explicit_script:
        script_path = Path(explicit_script).expanduser().resolve()
        return script_path.parents[1]

    candidate = Path(__file__).resolve().parents[2] / "openvla-oft"
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        "Could not infer OpenVLA-OFT root. Pass --openvla_oft_root or set OPENVLA_OFT_ROOT."
    )


def resolve_finetune_script(openvla_oft_root: Path, explicit_script: str | None) -> Path:
    if explicit_script:
        script_path = Path(explicit_script).expanduser().resolve()
    else:
        script_path = openvla_oft_root / "vla-scripts" / "finetune.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"OpenVLA-OFT fine-tuning script not found: {script_path}")
    return script_path


def resolve_torchrun(explicit_torchrun: str | None) -> str:
    if explicit_torchrun:
        return explicit_torchrun
    torchrun = shutil.which("torchrun")
    if torchrun:
        return torchrun
    raise FileNotFoundError("Could not find torchrun. Activate the OpenVLA-OFT environment first.")


def infer_tfds_dataset_from_rlds_dir(raw_path: str) -> tuple[Path, str]:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RLDS / TFDS path does not exist: {path}")

    if (path / "dataset_info.json").is_file():
        dataset_root = path.parent
        if dataset_root.name.replace(".", "").isdigit():
            dataset_root = dataset_root.parent
        return dataset_root.parent, dataset_root.name

    version_dirs = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "dataset_info.json").is_file()
    )
    if len(version_dirs) == 1:
        return path.parent, path.name
    if len(version_dirs) > 1:
        raise ValueError(f"Multiple TFDS versions found under {path}; pass the exact version directory.")

    nested_dataset_roots = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        child_versions = [
            grandchild for grandchild in child.iterdir()
            if grandchild.is_dir() and (grandchild / "dataset_info.json").is_file()
        ]
        if len(child_versions) == 1:
            nested_dataset_roots.append(child)
    if len(nested_dataset_roots) == 1:
        dataset_root = nested_dataset_roots[0]
        return dataset_root.parent, dataset_root.name

    raise FileNotFoundError(
        f"Could not infer TFDS dataset root from {path}. Pass --data_root_dir and --dataset_name explicitly."
    )


def resolve_dataset_args(args: argparse.Namespace) -> tuple[Path, str]:
    if args.rlds_dir:
        inferred_root, inferred_name = infer_tfds_dataset_from_rlds_dir(args.rlds_dir)
        data_root_dir = Path(args.data_root_dir).expanduser().resolve() if args.data_root_dir else inferred_root
        dataset_name = args.dataset_name or inferred_name
    else:
        if not args.data_root_dir or not args.dataset_name:
            raise ValueError("Pass either --rlds_dir or both --data_root_dir and --dataset_name.")
        data_root_dir = Path(args.data_root_dir).expanduser().resolve()
        dataset_name = args.dataset_name

    if not data_root_dir.exists():
        raise FileNotFoundError(f"data_root_dir does not exist: {data_root_dir}")
    return data_root_dir, dataset_name


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    openvla_oft_root = resolve_openvla_oft_root(args.openvla_oft_root, args.finetune_script)
    finetune_script = resolve_finetune_script(openvla_oft_root, args.finetune_script)
    data_root_dir, dataset_name = resolve_dataset_args(args)
    torchrun = resolve_torchrun(args.torchrun)

    command = [torchrun]
    if args.standalone:
        command.append("--standalone")
    command.extend(["--nnodes", str(args.nnodes)])
    command.extend(["--nproc-per-node", str(args.nproc_per_node)])
    if args.master_port:
        command.extend(["--master-port", str(args.master_port)])
    command.append(str(finetune_script))

    run_id_note = args.run_id_note or args.job_name
    run_root_dir = Path(args.run_root_dir).expanduser().resolve()

    append_arg(command, "vla_path", args.vla_path)
    append_arg(command, "data_root_dir", data_root_dir)
    append_arg(command, "dataset_name", dataset_name)
    append_arg(command, "run_root_dir", run_root_dir)
    append_arg(command, "run_id_note", run_id_note)
    append_arg(command, "run_id_override", args.run_id_override)
    append_arg(command, "use_l1_regression", args.use_l1_regression)
    append_arg(command, "use_diffusion", args.use_diffusion)
    append_arg(command, "use_film", args.use_film)
    append_arg(command, "num_images_in_input", args.num_images_in_input)
    append_arg(command, "use_proprio", args.use_proprio)
    append_arg(command, "batch_size", args.batch_size)
    append_arg(command, "learning_rate", args.learning_rate)
    append_arg(command, "lr_warmup_steps", args.lr_warmup_steps)
    append_arg(command, "num_steps_before_decay", args.num_steps_before_decay)
    append_arg(command, "grad_accumulation_steps", args.grad_accumulation_steps)
    append_arg(command, "max_steps", args.max_steps)
    append_arg(command, "save_freq", args.save_freq)
    append_arg(command, "save_latest_checkpoint_only", args.save_latest_checkpoint_only)
    append_arg(command, "resume", args.resume)
    append_arg(command, "resume_step", args.resume_step)
    append_arg(command, "image_aug", args.image_aug)
    append_arg(command, "shuffle_buffer_size", args.shuffle_buffer_size)
    append_arg(command, "use_val_set", args.use_val_set)
    append_arg(command, "val_freq", args.val_freq)
    append_arg(command, "val_time_limit", args.val_time_limit)
    append_arg(command, "use_lora", args.use_lora)
    append_arg(command, "lora_rank", args.lora_rank)
    append_arg(command, "lora_dropout", args.lora_dropout)
    append_arg(command, "merge_lora_during_training", args.merge_lora_during_training)
    append_arg(command, "wandb_entity", args.wandb_entity)
    append_arg(command, "wandb_project", args.wandb_project)
    append_arg(command, "wandb_log_freq", args.wandb_log_freq)

    command.extend(args.extra_arg)
    return command, openvla_oft_root


def main() -> None:
    args = parse_args()
    command, openvla_oft_root = build_command(args)

    print("Resolved OpenVLA-OFT training command:")
    print(f"cd {shlex.quote(str(openvla_oft_root))}")
    print(shlex.join(command))

    if args.dry_run:
        return

    try:
        subprocess.run(command, cwd=openvla_oft_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
