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

    def __init__(self, repo_ids: list[str], dataset_root: Path, dataset_kwargs: dict[str, object] | None = None) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._dataset_cls = LeRobotDataset
        self.repo_ids = list(repo_ids)
        self.root = Path(dataset_root).expanduser().resolve()
        self.dataset_kwargs = dict(dataset_kwargs or {})
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
        self._merge_meta_stats()
        self._patch_meta()

    def _load_dataset(self, dataset_index: int):
        dataset = self.datasets[dataset_index]
        if dataset is None:
            dataset_kwargs = filter_kwargs_for_callable(self._dataset_cls, self.dataset_kwargs)
            dataset = self._dataset_cls(
                repo_id=self.repo_ids[dataset_index],
                root=self.dataset_roots[dataset_index],
                **dataset_kwargs,
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

    def _merge_meta_stats(self) -> None:
        try:
            merged_stats = merge_weighted_stats([self._load_dataset(index) for index in range(len(self.repo_ids))])
        except ImportError as exc:
            print(f"WARNING: Could not merge per-shard normalization stats: {exc}")
            return

        if merged_stats is None:
            print("WARNING: Could not merge per-shard normalization stats; using first shard stats.")
            return

        try:
            self.meta.stats = merged_stats
            print(f"Merged normalization stats across {len(self.repo_ids)} local shards.")
        except AttributeError:
            print("WARNING: Dataset metadata stats are read-only; using first shard stats.")

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


def filter_kwargs_for_callable(callable_object, kwargs: dict[str, object]) -> dict[str, object]:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def import_lerobot_dataset_metadata_class():
    module_candidates = (
        "lerobot.datasets.dataset_metadata",
        "lerobot.datasets.lerobot_dataset",
    )
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        metadata_class = getattr(module, "LeRobotDatasetMetadata", None)
        if metadata_class is not None:
            return metadata_class
    raise ModuleNotFoundError("Could not import LeRobotDatasetMetadata from the installed LeRobot package.")


def make_image_transforms(image_transforms_cfg):
    if image_transforms_cfg is None or not getattr(image_transforms_cfg, "enable", False):
        return None

    module_candidates = ("lerobot.transforms", "lerobot.datasets.transforms")
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        image_transforms_class = getattr(module, "ImageTransforms", None)
        if image_transforms_class is not None:
            return image_transforms_class(image_transforms_cfg)

    print("WARNING: Could not import LeRobot ImageTransforms; continuing without image transforms.")
    return None


def make_local_shard_dataset_kwargs(cfg, repo_ids: list[str], dataset_root: Path) -> dict[str, object]:
    dataset_cfg = cfg.dataset
    dataset_kwargs = {
        "episodes": getattr(dataset_cfg, "episodes", None),
        "image_transforms": make_image_transforms(getattr(dataset_cfg, "image_transforms", None)),
        "revision": getattr(dataset_cfg, "revision", None),
        "video_backend": getattr(dataset_cfg, "video_backend", None),
        "tolerance_s": getattr(cfg, "tolerance_s", None),
    }

    dataset_kwargs = {key: value for key, value in dataset_kwargs.items() if value is not None}

    try:
        dataset_factory_module = importlib.import_module("lerobot.datasets.factory")
        resolve_delta_timestamps = getattr(dataset_factory_module, "resolve_delta_timestamps")
        metadata_class = import_lerobot_dataset_metadata_class()
        first_repo_id = repo_ids[0]
        first_root = resolve_local_dataset_root(dataset_root, first_repo_id)
        metadata_kwargs = {
            "repo_id": first_repo_id,
            "root": first_root,
            "revision": getattr(dataset_cfg, "revision", None),
        }
        metadata = metadata_class(**filter_kwargs_for_callable(metadata_class, metadata_kwargs))
        delta_timestamps = resolve_delta_timestamps(cfg.policy, metadata)
    except Exception as exc:
        print(f"WARNING: Could not resolve policy delta_timestamps for local shards: {exc}")
        delta_timestamps = None

    if delta_timestamps is not None:
        dataset_kwargs["delta_timestamps"] = delta_timestamps
        action_timestamps = delta_timestamps.get("action") if isinstance(delta_timestamps, dict) else None
        if action_timestamps is not None:
            print(
                "Resolved multi-shard action delta_timestamps: "
                f"len={len(action_timestamps)} first={action_timestamps[0]} last={action_timestamps[-1]}"
            )
        else:
            print(f"Resolved multi-shard delta_timestamps keys: {sorted(delta_timestamps)}")
    else:
        print("Resolved multi-shard delta_timestamps: None")

    return dataset_kwargs


def get_imagenet_stats():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to apply ImageNet camera stats.") from exc

    for module_name in ("lerobot.datasets.factory", "lerobot.utils.constants"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        imagenet_stats = getattr(module, "IMAGENET_STATS", None)
        if imagenet_stats is not None:
            return imagenet_stats

    return {
        "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32),
        "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32),
    }


def apply_imagenet_stats_if_requested(dataset, cfg) -> None:
    if not getattr(cfg.dataset, "use_imagenet_stats", False):
        return

    try:
        import torch
        imagenet_stats = get_imagenet_stats()
    except ImportError as exc:
        print(f"WARNING: Could not apply ImageNet camera stats: {exc}")
        return

    camera_keys = getattr(dataset.meta, "camera_keys", [])
    stats = getattr(dataset.meta, "stats", None)
    if not camera_keys or stats is None:
        return

    for key in camera_keys:
        if key not in stats:
            continue
        for stats_type, values in imagenet_stats.items():
            if isinstance(values, torch.Tensor):
                stats[key][stats_type] = values.detach().clone().to(dtype=torch.float32)
            else:
                stats[key][stats_type] = torch.tensor(values, dtype=torch.float32)
    print(f"Applied ImageNet camera stats to {len(camera_keys)} camera features.")


def patch_make_dataset_for_local_shards(train_module, repo_ids: list[str], dataset_root: Path) -> None:
    def _patched_make_dataset(cfg):
        dataset_kwargs = make_local_shard_dataset_kwargs(cfg, repo_ids, dataset_root)
        dataset = MultiLocalLeRobotDataset(repo_ids=repo_ids, dataset_root=dataset_root, dataset_kwargs=dataset_kwargs)
        apply_imagenet_stats_if_requested(dataset, cfg)
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


def trim_last_dim_pair(left, right):
    if (
        hasattr(left, "shape")
        and hasattr(right, "shape")
        and len(left.shape) > 0
        and len(right.shape) > 0
        and left.shape[-1] != right.shape[-1]
    ):
        target_length = min(int(left.shape[-1]), int(right.shape[-1]))
        left = left[..., :target_length]
        right = right[..., :target_length]
    return left, right


def collapse_short_action_window(actions, model_label: str, report_state: dict[str, bool]):
    if not hasattr(actions, "shape") or len(actions.shape) < 3:
        return actions

    shape = tuple(int(dim) for dim in actions.shape)
    action_like = shape[-1] <= 64
    short_window = action_like and len(shape) == 3 and shape[1] <= 8
    nested_short_window = action_like and len(shape) == 4 and shape[1] <= 8
    if not short_window and not nested_short_window:
        return actions

    collapsed_actions = actions[:, 0, ...]
    if not report_state["reported_action"]:
        print(f"Adjusted {model_label} action shape: {shape} -> {tuple(int(dim) for dim in collapsed_actions.shape)}")
        report_state["reported_action"] = True
    return collapsed_actions


def is_action_batch_key(key) -> bool:
    """Match plain or enum-like action keys while avoiding pad/mask fields."""
    key_value = getattr(key, "value", key)
    key_text = str(key_value).strip().strip("'\"").lower()
    if "mask" in key_text or "pad" in key_text:
        return False

    key_leaf = key_text.replace("]", "").split("[")[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1].strip("'\"")
    return key_leaf in {"action", "actions"}


def align_short_action_loss_pair(input_tensor, target_tensor, model_label: str, report_state: dict[str, bool]):
    """Prevent short action windows from broadcasting incorrectly at the PI flow loss."""
    if not hasattr(input_tensor, "shape") or not hasattr(target_tensor, "shape"):
        return input_tensor, target_tensor

    input_shape = tuple(int(dim) for dim in input_tensor.shape)
    target_shape = tuple(int(dim) for dim in target_tensor.shape)
    if abs(len(input_shape) - len(target_shape)) != 1:
        return input_tensor, target_tensor

    if len(input_shape) > len(target_shape):
        longer_shape = input_shape
        shorter_tensor, shorter_shape = target_tensor, target_shape
        shorter_is_input = False
    else:
        longer_shape = target_shape
        shorter_tensor, shorter_shape = input_tensor, input_shape
        shorter_is_input = True

    action_like = len(shorter_shape) >= 2 and len(longer_shape) == len(shorter_shape) + 1
    action_like = action_like and longer_shape[0] == shorter_shape[0]
    action_like = action_like and longer_shape[-1] == shorter_shape[-1]
    action_like = action_like and longer_shape[-1] <= 64 and longer_shape[1] <= 8
    if not action_like or not hasattr(shorter_tensor, "unsqueeze"):
        return input_tensor, target_tensor

    adjusted_shorter = shorter_tensor.unsqueeze(1)
    if not report_state["reported_loss"]:
        adjusted_shape = tuple(int(dim) for dim in adjusted_shorter.shape)
        print(
            f"Adjusted {model_label} action loss rank: "
            f"{shorter_shape} -> {adjusted_shape} against {longer_shape}"
        )
        report_state["reported_loss"] = True

    if shorter_is_input:
        return adjusted_shorter, target_tensor
    return input_tensor, adjusted_shorter


def patch_policy_action_loss_compat(policy_module, model_label: str) -> None:
    functional_module = getattr(policy_module, "F", None)
    if functional_module is None or getattr(functional_module, "_robocerebra_action_loss_compat", False):
        return

    mse_loss = getattr(functional_module, "mse_loss", None)
    if mse_loss is None:
        return

    class _FunctionalCompatProxy:
        _robocerebra_action_loss_compat = True

        def __init__(self, wrapped):
            self._wrapped = wrapped
            self._state = {"reported_loss": False}

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def mse_loss(self, input_tensor, target_tensor, *args, **kwargs):
            input_tensor, target_tensor = align_short_action_loss_pair(
                input_tensor, target_tensor, model_label, self._state
            )
            return self._wrapped.mse_loss(input_tensor, target_tensor, *args, **kwargs)

    policy_module.F = _FunctionalCompatProxy(functional_module)
    print(f"Applied {model_label} action-loss compatibility patch.")


def patch_policy_text_mask_compat(model_family: str) -> None:
    """Trim policy text padding when tokenizer ids and masks disagree."""
    model_label = "PI0.5" if model_family == "pi05" else model_family.upper()
    try:
        policy_module = importlib.import_module(f"lerobot.policies.{model_family}.modeling_{model_family}")
    except ModuleNotFoundError:
        return

    patch_policy_action_loss_compat(policy_module, model_label)

    for class_object in vars(policy_module).values():
        if not isinstance(class_object, type):
            continue

        forward = getattr(class_object, "forward", None)
        if forward is None or getattr(forward, "_robocerebra_batch_action_compat", False):
            continue

        try:
            parameter_names = list(inspect.signature(forward).parameters)
        except (TypeError, ValueError):
            continue
        if parameter_names[:2] != ["self", "batch"]:
            continue

        original_forward = forward
        state = {"reported_action": False}

        def _forward_with_batch_action_compat(
            self,
            batch,
            *args,
            __original_forward=original_forward,
            __model_label=model_label,
            __state=state,
            **kwargs,
        ):
            if isinstance(batch, dict):
                adjusted_batch = batch
                for batch_key, batch_value in batch.items():
                    if not is_action_batch_key(batch_key):
                        continue

                    adjusted_action = collapse_short_action_window(batch_value, __model_label, __state)
                    if adjusted_action is batch_value:
                        continue

                    if adjusted_batch is batch:
                        adjusted_batch = dict(batch)
                    adjusted_batch[batch_key] = adjusted_action
                batch = adjusted_batch
            return __original_forward(self, batch, *args, **kwargs)

        _forward_with_batch_action_compat._robocerebra_batch_action_compat = True
        setattr(class_object, "forward", _forward_with_batch_action_compat)

    patched_classes = []
    for class_name, class_object in vars(policy_module).items():
        if not isinstance(class_object, type):
            continue

        forward = getattr(class_object, "forward", None)
        if forward is None or getattr(forward, "_robocerebra_token_mask_compat", False):
            continue

        try:
            parameter_names = list(inspect.signature(forward).parameters)
        except (TypeError, ValueError):
            continue

        token_mask_pair = None
        for token_name, mask_name in (("tokens", "masks"), ("lang_tokens", "lang_masks")):
            if token_name in parameter_names and mask_name in parameter_names:
                token_mask_pair = (token_name, mask_name)
                break
        if parameter_names[:1] != ["self"] or token_mask_pair is None:
            continue

        original_forward = forward
        token_name, mask_name = token_mask_pair
        token_arg_index = parameter_names.index(token_name) - 1
        mask_arg_index = parameter_names.index(mask_name) - 1
        action_arg_index = None
        for action_name in ("actions", "action"):
            if action_name in parameter_names:
                action_arg_index = parameter_names.index(action_name) - 1
                break
        state = {"reported_action": False}

        def _forward_with_token_mask_compat(
            self,
            *args,
            __original_forward=original_forward,
            __token_arg_index=token_arg_index,
            __mask_arg_index=mask_arg_index,
            __action_arg_index=action_arg_index,
            __token_name=token_name,
            __mask_name=mask_name,
            __model_label=model_label,
            __state=state,
            **kwargs,
        ):
            args = list(args)
            if __token_name in kwargs and __mask_name in kwargs:
                kwargs[__token_name], kwargs[__mask_name] = trim_last_dim_pair(
                    kwargs[__token_name], kwargs[__mask_name]
                )
            elif __token_arg_index < len(args) and __mask_arg_index < len(args):
                args[__token_arg_index], args[__mask_arg_index] = trim_last_dim_pair(
                    args[__token_arg_index], args[__mask_arg_index]
                )
            if "actions" in kwargs:
                kwargs["actions"] = collapse_short_action_window(kwargs["actions"], __model_label, __state)
            elif "action" in kwargs:
                kwargs["action"] = collapse_short_action_window(kwargs["action"], __model_label, __state)
            elif __action_arg_index is not None and __action_arg_index < len(args):
                args[__action_arg_index] = collapse_short_action_window(
                    args[__action_arg_index], __model_label, __state
                )
            args = [collapse_short_action_window(arg, __model_label, __state) for arg in args]
            return __original_forward(self, *args, **kwargs)

        _forward_with_token_mask_compat._robocerebra_token_mask_compat = True
        setattr(class_object, "forward", _forward_with_token_mask_compat)
        patched_classes.append(class_name)

    if patched_classes:
        print(f"Applied {model_label} token/mask compatibility patch to: " + ", ".join(sorted(patched_classes)))
    else:
        print(f"WARNING: Could not locate a {model_label} model forward method for token/mask compatibility patch.")

    make_att_2d_masks = getattr(policy_module, "make_att_2d_masks", None)
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
            pad_masks, att_masks = trim_last_dim_pair(pad_masks, att_masks)

            if not state["reported"]:
                print(
                    f"Adjusted {model_label} attention mask lengths: "
                    f"pad={pad_length}, att={att_length}, target={int(pad_masks.shape[-1])}"
                )
                state["reported"] = True

        return make_att_2d_masks(pad_masks, att_masks)

    _make_att_2d_masks_with_token_mask_compat._robocerebra_token_mask_compat = True
    policy_module.make_att_2d_masks = _make_att_2d_masks_with_token_mask_compat
    print(f"Applied {model_label} 2D attention-mask compatibility patch.")


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
        model_family = canonicalize_model_family(args.model_family)
        if model_family in {"pi0", "pi05"}:
            patch_policy_text_mask_compat(model_family)
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
