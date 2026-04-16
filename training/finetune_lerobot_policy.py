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
import copy
import inspect
import importlib
import json
import shlex
import shutil
import subprocess
import sys
from bisect import bisect_right
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


def parse_dataset_repo_ids(raw_repo_id: str) -> list[str]:
    raw_repo_id = raw_repo_id.strip()
    if not raw_repo_id:
        raise ValueError("--dataset_repo_id cannot be empty.")

    if raw_repo_id.startswith("["):
        parsed = json.loads(raw_repo_id)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("--dataset_repo_id JSON must be a list of strings.")
        repo_ids = [item.strip() for item in parsed if item.strip()]
    else:
        repo_ids = [item.strip() for item in raw_repo_id.split(",") if item.strip()]

    if not repo_ids:
        raise ValueError("--dataset_repo_id did not contain any usable dataset repo ids.")
    return repo_ids


def resolve_local_dataset_root(base_root: Path, repo_id: str) -> Path:
    base_root = base_root.expanduser().resolve()
    candidates = [
        base_root,
        base_root / repo_id,
        base_root / repo_id.split("/")[-1],
    ]
    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    return base_root / repo_id


def read_lerobot_dataset_counts(dataset_root: Path) -> tuple[int | None, int | None]:
    info_path = dataset_root / "meta" / "info.json"
    total_frames = None
    total_episodes = None

    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for key in ("total_frames", "num_frames"):
            if key in info:
                total_frames = int(info[key])
                break
        for key in ("total_episodes", "num_episodes"):
            if key in info:
                total_episodes = int(info[key])
                break

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if episodes_path.is_file() and (total_frames is None or total_episodes is None):
        inferred_frames = 0
        inferred_episodes = 0
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                inferred_episodes += 1
                row = json.loads(line)
                for key in ("length", "num_frames", "episode_length"):
                    if key in row:
                        inferred_frames += int(row[key])
                        break
        if total_episodes is None:
            total_episodes = inferred_episodes
        if total_frames is None and inferred_frames > 0:
            total_frames = inferred_frames

    return total_frames, total_episodes


def import_lerobot_train_module():
    repo_lerobot_src = Path(__file__).resolve().parents[1] / "lerobot" / "src"
    if repo_lerobot_src.is_dir() and str(repo_lerobot_src) not in sys.path:
        sys.path.insert(0, str(repo_lerobot_src))

    for module_name in ("lerobot.scripts.lerobot_train", "lerobot.scripts.train"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        "Could not import LeRobot's training entrypoint. Install LeRobot or ensure `./lerobot/src` exists."
    )


def maybe_as_tensor(value):
    try:
        import torch
    except ImportError:  # pragma: no cover - LeRobot training requires torch.
        return None

    if isinstance(value, torch.Tensor):
        return value.detach().float()
    try:
        return torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None


def convert_tensor_like(tensor, template):
    import torch

    if isinstance(template, torch.Tensor):
        return tensor.to(device=template.device, dtype=template.dtype)
    return tensor.cpu().tolist()


def merge_weighted_stats(datasets):
    import torch

    stats_by_dataset = [getattr(getattr(dataset, "meta", None), "stats", None) for dataset in datasets]
    if not stats_by_dataset or any(stats is None for stats in stats_by_dataset):
        return None

    merged = copy.deepcopy(stats_by_dataset[0])
    weights = [float(getattr(dataset, "num_frames", len(dataset))) for dataset in datasets]
    total_weight = sum(weights)
    if total_weight <= 0:
        return merged

    for feature_key, feature_stats in list(merged.items()):
        if not isinstance(feature_stats, dict):
            continue

        source_feature_stats = [stats.get(feature_key, {}) for stats in stats_by_dataset]
        if all("mean" in stats and "std" in stats for stats in source_feature_stats):
            means = [maybe_as_tensor(stats["mean"]) for stats in source_feature_stats]
            stds = [maybe_as_tensor(stats["std"]) for stats in source_feature_stats]
            if all(mean is not None for mean in means) and all(std is not None for std in stds):
                weighted_mean = sum(weight * mean for weight, mean in zip(weights, means, strict=True)) / total_weight
                second_moment = sum(
                    weight * (std.square() + mean.square())
                    for weight, mean, std in zip(weights, means, stds, strict=True)
                ) / total_weight
                variance = torch.clamp(second_moment - weighted_mean.square(), min=1e-12)
                feature_stats["mean"] = convert_tensor_like(weighted_mean, source_feature_stats[0]["mean"])
                feature_stats["std"] = convert_tensor_like(torch.sqrt(variance), source_feature_stats[0]["std"])

        if all("min" in stats for stats in source_feature_stats):
            mins = [maybe_as_tensor(stats["min"]) for stats in source_feature_stats]
            if all(min_value is not None for min_value in mins):
                feature_stats["min"] = convert_tensor_like(torch.stack(mins).amin(dim=0), source_feature_stats[0]["min"])

        if all("max" in stats for stats in source_feature_stats):
            maxs = [maybe_as_tensor(stats["max"]) for stats in source_feature_stats]
            if all(max_value is not None for max_value in maxs):
                feature_stats["max"] = convert_tensor_like(torch.stack(maxs).amax(dim=0), source_feature_stats[0]["max"])

    return merged


class MultiLocalLeRobotDataset:
    """Present several local LeRobotDataset shards as one map-style dataset."""

    def __init__(self, repo_ids: list[str], dataset_root: Path) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._dataset_cls = LeRobotDataset
        self.repo_ids = list(repo_ids)
        self.root = Path(dataset_root).expanduser().resolve()
        self.datasets = [None] * len(self.repo_ids)
        self.dataset_roots = []
        self.shard_num_frames = []
        self.shard_num_episodes = []

        for repo_id in self.repo_ids:
            local_root = resolve_local_dataset_root(self.root, repo_id)
            self.dataset_roots.append(local_root)

            shard_frames, shard_episodes = read_lerobot_dataset_counts(local_root)
            self.shard_num_frames.append(shard_frames)
            self.shard_num_episodes.append(shard_episodes)

        if not self.repo_ids:
            raise ValueError("No local LeRobot shards were loaded.")

        first_dataset = self._load_dataset(0)
        if self.shard_num_frames[0] is None:
            self.shard_num_frames[0] = len(first_dataset)
        if self.shard_num_episodes[0] is None:
            self.shard_num_episodes[0] = int(getattr(first_dataset, "num_episodes", 0))

        missing_count_indices = [
            index for index, frame_count in enumerate(self.shard_num_frames) if frame_count is None
        ]
        if missing_count_indices:
            for index in missing_count_indices:
                dataset = self._load_dataset(index)
                self.shard_num_frames[index] = len(dataset)
                self.shard_num_episodes[index] = int(getattr(dataset, "num_episodes", 0))

        self.cumulative_sizes = []
        running_size = 0
        for shard_frames in self.shard_num_frames:
            running_size += int(shard_frames)
            self.cumulative_sizes.append(running_size)

        self.num_frames = sum(int(shard_frames) for shard_frames in self.shard_num_frames)
        self.num_episodes = sum(int(shard_episodes or 0) for shard_episodes in self.shard_num_episodes)
        try:
            self.meta = copy.deepcopy(first_dataset.meta)
        except Exception:
            # Some LeRobot metadata versions carry objects that do not deep-copy
            # cleanly. Reusing the first shard's metadata is still safe because
            # all shards were produced by the same exporter with the same schema.
            self.meta = first_dataset.meta
        self._patch_meta()

    def _load_dataset(self, dataset_index: int):
        dataset = self.datasets[dataset_index]
        if dataset is None:
            dataset = self._dataset_cls(
                repo_id=self.repo_ids[dataset_index],
                root=self.dataset_roots[dataset_index],
            )
            self.datasets[dataset_index] = dataset
        return dataset

    def _patch_meta(self) -> None:
        if hasattr(self.meta, "info") and isinstance(self.meta.info, dict):
            self.meta.info["total_frames"] = self.num_frames
            self.meta.info["total_episodes"] = self.num_episodes

        for attr_name, value in (("total_frames", self.num_frames), ("total_episodes", self.num_episodes)):
            if hasattr(self.meta, attr_name):
                try:
                    setattr(self.meta, attr_name, value)
                except AttributeError:
                    pass

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_size = self.cumulative_sizes[dataset_index - 1] if dataset_index > 0 else 0
        return self._load_dataset(dataset_index)[index - previous_size]


def patch_make_dataset_for_local_shards(train_module, repo_ids: list[str], dataset_root: Path) -> None:
    def _patched_make_dataset(cfg):
        del cfg
        dataset = MultiLocalLeRobotDataset(repo_ids=repo_ids, dataset_root=dataset_root)
        print(
            "Using multi-shard local LeRobot dataset: "
            f"{len(dataset.repo_ids)} shards, {dataset.num_episodes} episodes, {dataset.num_frames} frames"
        )
        return dataset

    train_module.make_dataset = _patched_make_dataset
    try:
        dataset_factory_module = importlib.import_module("lerobot.datasets.factory")
        dataset_factory_module.make_dataset = _patched_make_dataset
    except ModuleNotFoundError:
        pass


def patch_pi05_token_mask_compat() -> None:
    """Trim PI0.5 text padding when tokenizer ids and masks disagree."""
    try:
        modeling_pi05 = importlib.import_module("lerobot.policies.pi05.modeling_pi05")
    except ModuleNotFoundError:
        return

    patched_classes = []
    for class_name, class_object in vars(modeling_pi05).items():
        if not isinstance(class_object, type):
            continue

        forward = getattr(class_object, "forward", None)
        if forward is None or getattr(forward, "_robocerebra_token_mask_compat", False):
            continue

        try:
            parameter_names = list(inspect.signature(forward).parameters)
        except (TypeError, ValueError):
            continue
        if (
            len(parameter_names) < 6
            or parameter_names[0] != "self"
            or parameter_names[1:3] != ["images", "img_masks"]
            or parameter_names[5] != "actions"
        ):
            continue

        original_forward = forward

        def _forward_with_token_mask_compat(
            self,
            images,
            img_masks,
            tokens,
            masks,
            actions,
            *args,
            __original_forward=original_forward,
            **kwargs,
        ):
            if (
                hasattr(tokens, "shape")
                and hasattr(masks, "shape")
                and len(tokens.shape) > 0
                and len(masks.shape) > 0
                and tokens.shape[-1] != masks.shape[-1]
            ):
                token_length = int(tokens.shape[-1])
                mask_length = int(masks.shape[-1])
                target_length = min(token_length, mask_length)
                tokens = tokens[..., :target_length]
                masks = masks[..., :target_length]
            return __original_forward(self, images, img_masks, tokens, masks, actions, *args, **kwargs)

        _forward_with_token_mask_compat._robocerebra_token_mask_compat = True
        setattr(class_object, "forward", _forward_with_token_mask_compat)
        patched_classes.append(class_name)

    if patched_classes:
        print("Applied PI0.5 token/mask compatibility patch to: " + ", ".join(sorted(patched_classes)))
    else:
        print("WARNING: Could not locate a PI0.5 model forward method for token/mask compatibility patch.")

    make_att_2d_masks = getattr(modeling_pi05, "make_att_2d_masks", None)
    if make_att_2d_masks is None or getattr(make_att_2d_masks, "_robocerebra_token_mask_compat", False):
        return

    state = {"reported": False}

    def _make_att_2d_masks_with_token_mask_compat(pad_masks, att_masks):
        if (
            att_masks is not None
            and hasattr(pad_masks, "shape")
            and hasattr(att_masks, "shape")
            and len(pad_masks.shape) > 0
            and len(att_masks.shape) > 0
            and pad_masks.shape[-1] != att_masks.shape[-1]
        ):
            pad_length = int(pad_masks.shape[-1])
            att_length = int(att_masks.shape[-1])
            target_length = min(pad_length, att_length)
            pad_masks = pad_masks[..., :target_length]
            att_masks = att_masks[..., :target_length]

            if not state["reported"]:
                print(
                    "Adjusted PI0.5 attention mask lengths: "
                    f"pad={pad_length}, att={att_length}, target={target_length}"
                )
                state["reported"] = True

        return make_att_2d_masks(pad_masks, att_masks)

    _make_att_2d_masks_with_token_mask_compat._robocerebra_token_mask_compat = True
    modeling_pi05.make_att_2d_masks = _make_att_2d_masks_with_token_mask_compat
    print("Applied PI0.5 2D attention-mask compatibility patch.")


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
    repo_ids = parse_dataset_repo_ids(args.dataset_repo_id)

    command_args = args
    if len(repo_ids) > 1:
        command_args = copy.copy(args)
        command_args.dataset_repo_id = repo_ids[0]
        command_args.dataset_root = str(resolve_local_dataset_root(Path(args.dataset_root), repo_ids[0]))

    command = build_command(command_args)
    print("Resolved LeRobot training command:")
    print(shlex.join(command))
    if len(repo_ids) > 1:
        print("Multi-shard local datasets:")
        for repo_id in repo_ids:
            print(f"  - {repo_id}")

    if args.dry_run:
        return

    if len(repo_ids) > 1:
        train_module = import_lerobot_train_module()
        patch_make_dataset_for_local_shards(train_module, repo_ids, Path(args.dataset_root))
        if canonicalize_model_family(args.model_family) == "pi05":
            patch_pi05_token_mask_compat()
        sys.argv = command
        train_module.main()
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
