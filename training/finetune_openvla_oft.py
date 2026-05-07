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
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_VLA_PATH = "openvla/openvla-7b"
DEFAULT_JOB_NAME = "robocerebra_openvla_oft_finetune"
DEFAULT_ROBOCEREBRA_MIXTURE_NAME = "robocerebra_train_full"

ROBOCEREBRA_TRAINSET_PRESETS: dict[str, tuple[str, ...]] = {
    "full": (
        "RoboCerebra_trainset_coffee_table_p1p2_rlds/homerobo_trainset_p1p2",
        "RoboCerebra_trainset_coffee_table_p3_rlds/homerobo_trainset_p1p2",
        "RoboCerebra_trainset_kitchen_table_p1_rlds/homerobo_trainset_p1p2",
        "RoboCerebra_trainset_study_table_p1_rlds/homerobo_trainset_p1p2",
    ),
    "coffee_table_p1p2": (
        "RoboCerebra_trainset_coffee_table_p1p2_rlds/homerobo_trainset_p1p2",
    ),
    "coffee_table_p3": (
        "RoboCerebra_trainset_coffee_table_p3_rlds/homerobo_trainset_p1p2",
    ),
    "kitchen_table_p1": (
        "RoboCerebra_trainset_kitchen_table_p1_rlds/homerobo_trainset_p1p2",
    ),
    "study_table_p1": (
        "RoboCerebra_trainset_study_table_p1_rlds/homerobo_trainset_p1p2",
    ),
}


@dataclass(frozen=True)
class TFDSDatasetRef:
    data_root_dir: Path
    dataset_name: str
    dataset_root: Path


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
        action="append",
        default=[],
        help=(
            "Path to a TFDS dataset directory. Repeat this flag to match the four-way RoboCerebra "
            "training data used for the PI conversion. Pass either the dataset root or its version directory."
        ),
    )
    parser.add_argument(
        "--robocerebra_data_root",
        default=None,
        help=(
            "Root containing RoboCerebra_trainset_*_rlds directories. Used with "
            "--robocerebra_trainset_preset."
        ),
    )
    parser.add_argument(
        "--robocerebra_trainset_preset",
        choices=sorted(ROBOCEREBRA_TRAINSET_PRESETS),
        default=None,
        help=(
            "Expand to the same RLDS / TFDS subsets used by the PI conversion. "
            "Use 'full' for coffee_table_p1p2, coffee_table_p3, kitchen_table_p1, and study_table_p1."
        ),
    )
    parser.add_argument(
        "--stage_data_root_dir",
        default=None,
        help=(
            "Directory where --rlds_dir datasets are symlinked under unique TFDS names so OpenVLA-OFT "
            "can load them from one data_root_dir. Use a local path such as /tmp/... on clusters where "
            "/gs/... paths are interpreted as GCS. Defaults to $ROBOCEREBRA_TFDS_STAGE_ROOT/<job_name>, "
            "or /tmp/<user>_robocerebra_tfds/<job_name>."
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
    parser.add_argument(
        "--register_robocerebra_dataset",
        type=parse_bool,
        default=True,
        help=(
            "Run OpenVLA-OFT through a generated wrapper that registers RoboCerebra's LIBERO-style "
            "TFDS feature mapping before training starts."
        ),
    )
    parser.add_argument(
        "--robocerebra_mixture_name",
        default=DEFAULT_ROBOCEREBRA_MIXTURE_NAME,
        help="OpenVLA-OFT mixture name used when repeated --rlds_dir inputs are provided.",
    )
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


def infer_tfds_dataset_from_rlds_dir(raw_path: str) -> TFDSDatasetRef:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RLDS / TFDS path does not exist: {path}")

    if (path / "dataset_info.json").is_file():
        dataset_root = path.parent
        if dataset_root.name.replace(".", "").isdigit():
            dataset_root = dataset_root.parent
        return TFDSDatasetRef(
            data_root_dir=dataset_root.parent,
            dataset_name=dataset_root.name,
            dataset_root=dataset_root,
        )

    version_dirs = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "dataset_info.json").is_file()
    )
    if len(version_dirs) == 1:
        return TFDSDatasetRef(data_root_dir=path.parent, dataset_name=path.name, dataset_root=path)
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
        return TFDSDatasetRef(
            data_root_dir=dataset_root.parent,
            dataset_name=dataset_root.name,
            dataset_root=dataset_root,
        )

    raise FileNotFoundError(
        f"Could not infer TFDS dataset root from {path}. Pass --data_root_dir and --dataset_name explicitly."
    )


def expand_rlds_dirs(args: argparse.Namespace) -> list[str]:
    rlds_dirs = list(args.rlds_dir or [])
    if args.robocerebra_trainset_preset:
        if not args.robocerebra_data_root:
            raise ValueError("--robocerebra_trainset_preset requires --robocerebra_data_root.")
        data_root = Path(args.robocerebra_data_root).expanduser().resolve()
        rlds_dirs.extend(
            str(data_root / rel_path)
            for rel_path in ROBOCEREBRA_TRAINSET_PRESETS[args.robocerebra_trainset_preset]
        )
    return rlds_dirs


def sanitize_tfds_name(value: str) -> str:
    """Return a filesystem- and TFDS-friendly dataset alias."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("Cannot create an empty TFDS dataset alias.")
    if value[0].isdigit():
        value = f"dataset_{value}"
    return value


def infer_robocerebra_dataset_alias(ref: TFDSDatasetRef, index: int) -> str:
    """Infer a stable alias for staged RoboCerebra trainset shards."""
    parent_name = ref.dataset_root.parent.name
    prefix = "RoboCerebra_trainset_"
    suffix = "_rlds"
    if parent_name.startswith(prefix) and parent_name.endswith(suffix):
        shard_name = parent_name[len(prefix):-len(suffix)]
        return sanitize_tfds_name(f"robocerebra_{shard_name}")
    return sanitize_tfds_name(f"{ref.dataset_name}_{index}")


def unique_tfds_aliases(dataset_refs: list[TFDSDatasetRef]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    duplicate_source_names = len({ref.dataset_name for ref in dataset_refs}) != len(dataset_refs)
    for index, ref in enumerate(dataset_refs):
        alias = infer_robocerebra_dataset_alias(ref, index) if duplicate_source_names else ref.dataset_name
        alias = sanitize_tfds_name(alias)
        base_alias = alias
        suffix = 1
        while alias in seen:
            suffix += 1
            alias = sanitize_tfds_name(f"{base_alias}_{suffix}")
        aliases.append(alias)
        seen.add(alias)
    return aliases


def default_stage_root(args: argparse.Namespace) -> Path:
    raw_root = os.environ.get("ROBOCEREBRA_TFDS_STAGE_ROOT")
    if raw_root:
        return Path(raw_root).expanduser().resolve() / args.job_name
    user = os.environ.get("USER") or "user"
    return Path("/tmp") / f"{user}_robocerebra_tfds" / args.job_name


def stage_tfds_dataset_roots(
    dataset_refs: list[TFDSDatasetRef], stage_root: Path, dry_run: bool
) -> tuple[Path, list[str]]:
    dataset_aliases = unique_tfds_aliases(dataset_refs)

    if dry_run:
        return stage_root, dataset_aliases

    stage_root.mkdir(parents=True, exist_ok=True)
    for ref, dataset_alias in zip(dataset_refs, dataset_aliases):
        link_path = stage_root / dataset_alias
        if link_path.exists() or link_path.is_symlink():
            try:
                if link_path.resolve() == ref.dataset_root.resolve():
                    continue
            except FileNotFoundError:
                pass
            raise FileExistsError(
                f"Cannot stage TFDS dataset {dataset_alias}: {link_path} already exists "
                f"and does not point to {ref.dataset_root}."
            )
        link_path.symlink_to(ref.dataset_root, target_is_directory=True)
    return stage_root, dataset_aliases


def resolve_dataset_args(args: argparse.Namespace) -> tuple[Path, str, list[str]]:
    expanded_rlds_dirs = expand_rlds_dirs(args)
    if expanded_rlds_dirs:
        dataset_refs = [infer_tfds_dataset_from_rlds_dir(path_str) for path_str in expanded_rlds_dirs]
        if args.stage_data_root_dir:
            stage_root = Path(args.stage_data_root_dir).expanduser().resolve()
            data_root_dir, registered_dataset_names = stage_tfds_dataset_roots(
                dataset_refs, stage_root, args.dry_run
            )
            dataset_name = args.dataset_name or (
                registered_dataset_names[0] if len(registered_dataset_names) == 1 else args.robocerebra_mixture_name
            )
        elif len(dataset_refs) == 1:
            data_root_dir = (
                Path(args.data_root_dir).expanduser().resolve()
                if args.data_root_dir
                else dataset_refs[0].data_root_dir
            )
            dataset_name = args.dataset_name or dataset_refs[0].dataset_name
            registered_dataset_names = [dataset_name]
        else:
            if args.data_root_dir:
                data_root_dir = Path(args.data_root_dir).expanduser().resolve()
                registered_dataset_names = unique_tfds_aliases(dataset_refs)
            else:
                stage_root = (
                    Path(args.stage_data_root_dir).expanduser().resolve()
                    if args.stage_data_root_dir
                    else default_stage_root(args)
                )
                data_root_dir, registered_dataset_names = stage_tfds_dataset_roots(
                    dataset_refs, stage_root, args.dry_run
                )
            dataset_name = args.dataset_name or args.robocerebra_mixture_name
    else:
        if not args.data_root_dir or not args.dataset_name:
            raise ValueError(
                "Pass --rlds_dir, --robocerebra_trainset_preset, or both --data_root_dir and --dataset_name."
            )
        data_root_dir = Path(args.data_root_dir).expanduser().resolve()
        dataset_name = args.dataset_name
        registered_dataset_names = [dataset_name]

    if not data_root_dir.exists():
        if args.dry_run:
            print(f"Warning: data_root_dir does not exist yet: {data_root_dir}")
        else:
            raise FileNotFoundError(f"data_root_dir does not exist: {data_root_dir}")
    return data_root_dir, dataset_name, registered_dataset_names


def write_robocerebra_openvla_wrapper(
    finetune_script: Path,
    openvla_oft_root: Path,
    run_root_dir: Path,
    job_name: str,
    dataset_names: list[str],
    mixture_name: str,
) -> Path:
    wrapper_dir = run_root_dir / "robocerebra_openvla_wrappers"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / f"{job_name}_finetune.py"
    payload = {
        "finetune_script": str(finetune_script),
        "dataset_names": dataset_names,
        "mixture_name": mixture_name,
    }
    payload_json = json.dumps(payload, indent=2)
    wrapper_source = f'''#!/usr/bin/env python3
"""Generated RoboCerebra OpenVLA-OFT launcher wrapper."""

import importlib
import json
import runpy
import sys


PAYLOAD = json.loads({payload_json!r})


def patch_robocerebra_dataset_registries() -> None:
    configs = importlib.import_module("prismatic.vla.datasets.rlds.oxe.configs")
    mixtures = importlib.import_module("prismatic.vla.datasets.rlds.oxe.mixtures")
    transforms = importlib.import_module("prismatic.vla.datasets.rlds.oxe.transforms")
    oxe = importlib.import_module("prismatic.vla.datasets.rlds.oxe")

    dataset_config = {{
        "image_obs_keys": {{"primary": "image", "secondary": None, "wrist": "wrist_image"}},
        "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
        "state_obs_keys": ["EEF_state", "gripper_state"],
        "state_encoding": configs.StateEncoding.POS_EULER,
        "action_encoding": configs.ActionEncoding.EEF_POS,
    }}

    dataset_names = list(PAYLOAD["dataset_names"])
    for dataset_name in dataset_names:
        configs.OXE_DATASET_CONFIGS[dataset_name] = {{
            key: value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value
            for key, value in dataset_config.items()
        }}
        transforms.OXE_STANDARDIZATION_TRANSFORMS[dataset_name] = transforms.libero_dataset_transform

    mixture_name = PAYLOAD["mixture_name"]
    mixtures.OXE_NAMED_MIXTURES[mixture_name] = [(dataset_name, 1.0) for dataset_name in dataset_names]
    oxe.OXE_NAMED_MIXTURES = mixtures.OXE_NAMED_MIXTURES


patch_robocerebra_dataset_registries()
sys.argv[0] = PAYLOAD["finetune_script"]
runpy.run_path(PAYLOAD["finetune_script"], run_name="__main__")
'''
    wrapper_path.write_text(wrapper_source, encoding="utf-8")
    wrapper_path.chmod(0o755)
    if wrapper_path.is_relative_to(openvla_oft_root):
        return wrapper_path.relative_to(openvla_oft_root)
    return wrapper_path


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    openvla_oft_root = resolve_openvla_oft_root(args.openvla_oft_root, args.finetune_script)
    finetune_script = resolve_finetune_script(openvla_oft_root, args.finetune_script)
    run_root_dir = Path(args.run_root_dir).expanduser().resolve()
    data_root_dir, dataset_name, registered_dataset_names = resolve_dataset_args(args)
    torchrun = resolve_torchrun(args.torchrun)
    script_to_launch = finetune_script
    if args.register_robocerebra_dataset:
        script_to_launch = write_robocerebra_openvla_wrapper(
            finetune_script=finetune_script,
            openvla_oft_root=openvla_oft_root,
            run_root_dir=run_root_dir,
            job_name=args.job_name,
            dataset_names=registered_dataset_names,
            mixture_name=args.robocerebra_mixture_name,
        )

    command = [torchrun]
    if args.standalone:
        command.append("--standalone")
    command.extend(["--nnodes", str(args.nnodes)])
    command.extend(["--nproc-per-node", str(args.nproc_per_node)])
    if args.master_port:
        command.extend(["--master-port", str(args.master_port)])
    command.append(str(script_to_launch))

    run_id_note = args.run_id_note or args.job_name

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
    print(shlex.join(command), flush=True)

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
