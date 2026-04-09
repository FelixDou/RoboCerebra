#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch LeRobot pi0 / pi0.5 fine-tuning directly from local RLDS / TFDS exports.

This avoids the expensive RLDS -> LeRobot conversion step by patching LeRobot's
dataset factory at runtime with a streaming PyTorch dataset backed by TFDS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import random
import shlex
import shutil
import sys
import time
from types import SimpleNamespace


os.environ["NO_GCE_CHECK"] = "true"

try:
    from tensorflow_datasets.core.utils import gcs_utils

    gcs_utils._is_gcs_disabled = True
except ImportError:  # pragma: no cover - optional depending on tfds version
    pass

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("This wrapper requires numpy.") from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("This wrapper requires tensorflow.") from exc

try:
    import tensorflow_datasets as tfds
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("This wrapper requires tensorflow-datasets.") from exc

try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("This wrapper requires torch.") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


DEFAULT_PRETRAINED_PATHS = {
    "pi0": "lerobot/pi0_base",
    "pi05": "lerobot/pi05_base",
}

IMAGENET_STATS = {
    "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32),
    "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32),
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
        description="Launch LeRobot pi0 / pi0.5 fine-tuning directly from local RLDS / TFDS exports."
    )
    parser.add_argument(
        "--model_family",
        required=True,
        choices=("pi0", "pi05", "pi-05", "pi-0.5", "pi_zero", "pi_zero_five", "pi-zero", "pi-zero-five"),
        help="Policy family to fine-tune.",
    )
    parser.add_argument(
        "--rlds_dir",
        action="append",
        required=True,
        help=(
            "Path to a local TFDS export directory. Repeat for multiple datasets. "
            "You may pass either the dataset root (containing `1.0.0/`) or the version directory itself."
        ),
    )
    parser.add_argument(
        "--dataset_repo_id",
        default="robocerebra/pi_train_rlds",
        help="Synthetic dataset repo id used for LeRobot config plumbing.",
    )
    parser.add_argument(
        "--dataset_root",
        default="./rlds_dataset_builder/rlds_direct_dataset",
        help="Synthetic dataset root used for LeRobot config plumbing.",
    )
    parser.add_argument(
        "--dataset_fps",
        type=int,
        default=10,
        help="Dataset fps metadata passed to the policy stack. Defaults to 10 to match the LeRobot LIBERO setup.",
    )
    parser.add_argument(
        "--metadata_cache",
        default=None,
        help="Optional path to cache scanned RLDS metadata. Defaults to a hashed file under `training/.cache/`.",
    )
    parser.add_argument(
        "--force_recompute_metadata",
        action="store_true",
        help="Ignore any existing metadata cache and rescan the RLDS builders.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes used from the RLDS sources. Useful for smoke tests.",
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
        help="Training job name. Defaults to robocerebra_<model_family>_rlds_finetune.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Parent output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--keep_only_last_checkpoint_on_success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After a successful training run, delete all checkpoint directories except the last one. "
            "Useful for saving disk space once the run finishes."
        ),
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
        "--num_workers",
        type=int,
        default=2,
        help="PyTorch DataLoader workers used to stream RLDS episodes.",
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
        "--extra_arg",
        action="append",
        default=[],
        help="Extra raw argument appended verbatim to the underlying LeRobot training invocation. Repeatable.",
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


def default_policy_repo_id(job_name: str) -> str:
    repo_suffix = job_name.removeprefix("robocerebra_")
    return f"robocerebra/{repo_suffix}"


def resolve_builder_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RLDS path does not exist: {path}")

    if (path / "dataset_info.json").is_file():
        return path

    version_dirs = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "dataset_info.json").is_file()
    )
    if len(version_dirs) == 1:
        return version_dirs[0]
    if len(version_dirs) > 1:
        raise ValueError(
            f"Multiple TFDS version directories found under {path}. "
            "Please pass the specific version directory you want to use."
        )

    nested_dataset_roots = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        child_version_dirs = sorted(
            grandchild
            for grandchild in child.iterdir()
            if grandchild.is_dir() and (grandchild / "dataset_info.json").is_file()
        )
        if len(child_version_dirs) == 1:
            nested_dataset_roots.append(child_version_dirs[0])
        elif len(child_version_dirs) > 1:
            raise ValueError(
                f"Multiple TFDS version directories found under nested dataset root {child}. "
                "Please pass the specific version directory you want to use."
            )

    if len(nested_dataset_roots) == 1:
        return nested_dataset_roots[0]
    if len(nested_dataset_roots) > 1:
        raise ValueError(
            f"Multiple nested TFDS dataset roots found under {path}. "
            "Please pass the specific dataset directory you want to use."
        )

    raise FileNotFoundError(
        f"Could not find a TFDS builder directory under {path}. "
        "Expected either `dataset_info.json` directly, a single child like `1.0.0/`, "
        "or a single nested dataset root containing `1.0.0/`."
    )


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "numpy"):
        value = value.numpy()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def disable_tensorflow_gpu() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def builder_dir_to_tfds_uri(builder_dir: Path) -> str:
    """Force TFDS to treat cluster paths like /gs/bs/... as local filesystem paths."""
    resolved = builder_dir.expanduser().resolve()
    try:
        relative = os.path.relpath(resolved, Path.cwd().resolve())
    except ValueError:
        return str(resolved)

    if not relative.startswith("."):
        relative = f"./{relative}"
    return relative


def wait_for_metadata_cache(cache_path: Path, timeout_s: int = 7200, poll_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if cache_path.exists():
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for RLDS metadata cache: {cache_path}")


def _cardinality_to_int(steps_ds) -> int:
    cardinality = tf.data.experimental.cardinality(steps_ds)
    if hasattr(cardinality, "numpy"):
        cardinality = int(cardinality.numpy())
    if cardinality >= 0:
        return int(cardinality)
    return sum(1 for _ in steps_ds)


def compute_default_cache_path(builder_dirs: list[Path], fps: int, max_episodes: int | None) -> Path:
    payload = json.dumps(
        {"builder_dirs": [str(path) for path in builder_dirs], "fps": fps, "max_episodes": max_episodes},
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"rlds_metadata_{digest}.json"


def _make_feature_spec(image_shape: tuple[int, int, int], state_dim: int, action_dim: int) -> dict[str, dict]:
    return {
        "observation.images.image": {
            "dtype": "image",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_image": {
            "dtype": "image",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_rx",
                "eef_ry",
                "eef_rz",
                "gripper_left",
                "gripper_right",
            ][:state_dim],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_rx",
                "eef_ry",
                "eef_rz",
                "gripper",
            ][:action_dim],
        },
    }


def _normalize_feature_spec(features: dict[str, dict]) -> dict[str, dict]:
    normalized = {}
    for key, feature in features.items():
        normalized_feature = dict(feature)
        shape = normalized_feature.get("shape")
        if isinstance(shape, list):
            normalized_feature["shape"] = tuple(shape)
        normalized[key] = normalized_feature
    return normalized


class RLDSMetadata(SimpleNamespace):
    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] in {"image", "video"}]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def task_to_index(self) -> dict[str, int]:
        return {row["task"]: int(row["task_index"]) for row in self.tasks}

    @property
    def index_to_task(self) -> dict[int, str]:
        return {int(row["task_index"]): row["task"] for row in self.tasks}


def _build_metadata_from_payload(payload: dict) -> RLDSMetadata:
    info = dict(payload["info"])
    info["features"] = _normalize_feature_spec(info["features"])
    stats = {}
    for key, per_feature in payload["stats"].items():
        stats[key] = {
            stat_name: torch.tensor(values, dtype=torch.float32) for stat_name, values in per_feature.items()
        }
    return RLDSMetadata(
        info=info,
        stats=stats,
        episodes=payload["episodes"],
        tasks=payload["tasks"],
        robot_type=payload["info"].get("robot_type", "panda"),
    )


def scan_rlds_metadata(
    builder_dirs: list[Path],
    fps: int,
    cache_path: Path,
    force_recompute: bool = False,
    max_episodes: int | None = None,
) -> RLDSMetadata:
    if cache_path.exists() and not force_recompute:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        logging.info("Loaded RLDS metadata cache from %s", cache_path)
        return _build_metadata_from_payload(payload)

    logging.info("Scanning RLDS builders to compute dataset metadata and normalization stats.")

    total_episodes = 0
    total_frames = 0
    task_to_index: dict[str, int] = {}
    episode_rows: list[dict[str, int | str]] = []

    state_sum = None
    state_sumsq = None
    action_sum = None
    action_sumsq = None
    image_shape = None
    state_dim = None
    action_dim = None

    stop_early = False
    for builder_dir in builder_dirs:
        builder = tfds.builder_from_directory(builder_dir=builder_dir_to_tfds_uri(builder_dir))
        ds = builder.as_dataset(split="train", shuffle_files=False)
        total_examples = getattr(builder.info.splits["train"], "num_examples", None)
        progress = ds
        if tqdm is not None:
            progress = tqdm(ds, total=total_examples, desc=f"Scanning {builder.info.name}", unit="episode")

        for episode in progress:
            steps_ds = episode["steps"]
            episode_length = _cardinality_to_int(steps_ds)
            if episode_length <= 0:
                continue

            task = None
            for step_idx, step in enumerate(steps_ds):
                if task is None:
                    task = decode_text(step["language_instruction"]).strip()

                state = np.asarray(step["observation"]["state"].numpy(), dtype=np.float64)
                action = np.asarray(step["action"].numpy(), dtype=np.float64)

                if image_shape is None:
                    image = np.asarray(step["observation"]["image"].numpy(), dtype=np.uint8)
                    image_shape = tuple(image.shape)
                    state_dim = int(state.shape[-1])
                    action_dim = int(action.shape[-1])
                    state_sum = np.zeros(state_dim, dtype=np.float64)
                    state_sumsq = np.zeros(state_dim, dtype=np.float64)
                    action_sum = np.zeros(action_dim, dtype=np.float64)
                    action_sumsq = np.zeros(action_dim, dtype=np.float64)

                state_sum += state
                state_sumsq += state * state
                action_sum += action
                action_sumsq += action * action

            task = task or ""
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)

            episode_rows.append(
                {
                    "episode_index": total_episodes,
                    "length": episode_length,
                    "task": task,
                    "task_index": task_to_index[task],
                    "dataset_from_index": total_frames,
                    "dataset_to_index": total_frames + episode_length,
                }
            )

            total_episodes += 1
            total_frames += episode_length

            if max_episodes is not None and total_episodes >= max_episodes:
                stop_early = True
                break

        if stop_early:
            break

    if total_frames == 0 or image_shape is None or state_dim is None or action_dim is None:
        raise RuntimeError("No RLDS frames were found. Please verify the provided TFDS export paths.")

    state_mean = state_sum / total_frames
    action_mean = action_sum / total_frames
    state_var = np.maximum(state_sumsq / total_frames - state_mean * state_mean, 1e-12)
    action_var = np.maximum(action_sumsq / total_frames - action_mean * action_mean, 1e-12)

    info = {
        "codebase_version": "rlds-direct",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "fps": fps,
        "features": _make_feature_spec(image_shape, state_dim, action_dim),
    }
    stats = {
        "observation.state": {
            "mean": state_mean.astype(np.float32).tolist(),
            "std": np.sqrt(state_var).astype(np.float32).tolist(),
        },
        "action": {
            "mean": action_mean.astype(np.float32).tolist(),
            "std": np.sqrt(action_var).astype(np.float32).tolist(),
        },
    }
    payload = {
        "info": info,
        "stats": stats,
        "episodes": episode_rows,
        "tasks": [{"task": task, "task_index": task_index} for task, task_index in sorted(task_to_index.items(), key=lambda kv: kv[1])],
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Saved RLDS metadata cache to %s", cache_path)
    return _build_metadata_from_payload(payload)


class RLDSStreamingDataset(IterableDataset):
    def __init__(
        self,
        builder_dirs: list[Path],
        metadata: RLDSMetadata,
        action_chunk_size: int,
        max_episodes: int | None = None,
    ) -> None:
        super().__init__()
        self.builder_dirs = list(builder_dirs)
        self.meta = metadata
        self.num_frames = metadata.total_frames
        self.num_episodes = metadata.total_episodes
        self.action_chunk_size = action_chunk_size
        self.max_episodes = max_episodes
        self._default_task_index = 0

    def __len__(self) -> int:
        return self.num_frames

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        total_shards = max(1, world_size * num_workers)
        shard_index = rank * num_workers + worker_id

        builder_dirs = list(self.builder_dirs)
        random.shuffle(builder_dirs)
        emitted_episodes = 0

        for builder_dir in builder_dirs:
            builder = tfds.builder_from_directory(builder_dir=builder_dir_to_tfds_uri(builder_dir))
            ds = builder.as_dataset(split="train", shuffle_files=True)
            if total_shards > 1:
                ds = ds.shard(total_shards, shard_index)

            for episode in ds:
                yield from self._yield_episode_samples(episode)
                emitted_episodes += 1
                if self.max_episodes is not None and emitted_episodes >= self.max_episodes:
                    return

    def _yield_episode_samples(self, episode):
        images: list[np.ndarray] = []
        wrist_images: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        task = ""

        for step in episode["steps"]:
            observation = step["observation"]
            images.append(np.asarray(observation["image"].numpy(), dtype=np.uint8))
            wrist_images.append(np.asarray(observation["wrist_image"].numpy(), dtype=np.uint8))
            states.append(np.asarray(observation["state"].numpy(), dtype=np.float32))
            actions.append(np.asarray(step["action"].numpy(), dtype=np.float32))
            if not task:
                task = decode_text(step["language_instruction"]).strip()

        if not actions:
            return

        action_dim = actions[0].shape[-1]
        zero_action = np.zeros((self.action_chunk_size, action_dim), dtype=np.float32)

        for step_idx in range(len(actions)):
            valid = min(self.action_chunk_size, len(actions) - step_idx)
            action_chunk = zero_action.copy()
            if valid > 0:
                action_chunk[:valid] = np.stack(actions[step_idx : step_idx + valid], axis=0)
            action_is_pad = np.ones(self.action_chunk_size, dtype=bool)
            action_is_pad[:valid] = False

            yield {
                "observation.images.image": torch.from_numpy(np.moveaxis(images[step_idx], -1, 0)).float() / 255.0,
                "observation.images.wrist_image": torch.from_numpy(np.moveaxis(wrist_images[step_idx], -1, 0)).float() / 255.0,
                "observation.state": torch.from_numpy(states[step_idx]).float(),
                "action": torch.from_numpy(action_chunk).float(),
                "action_is_pad": torch.from_numpy(action_is_pad),
                "task": torch.tensor(
                    self.meta.task_to_index.get(task, self._default_task_index),
                    dtype=torch.int64,
                ),
                "task_index": torch.tensor(
                    self.meta.task_to_index.get(task, self._default_task_index),
                    dtype=torch.int64,
                ),
            }


def apply_imagenet_stats_if_requested(cfg, metadata: RLDSMetadata) -> None:
    if not getattr(cfg.dataset, "use_imagenet_stats", False):
        return
    for key in metadata.camera_keys:
        metadata.stats.setdefault(key, {})
        metadata.stats[key]["mean"] = IMAGENET_STATS["mean"].clone()
        metadata.stats[key]["std"] = IMAGENET_STATS["std"].clone()


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


def patch_pi0_image_feature_compat() -> None:
    try:
        pi0_module = importlib.import_module("lerobot.policies.pi0.modeling_pi0")
    except ModuleNotFoundError:
        return

    model_cls = getattr(pi0_module, "PaliGemmaWithExpertModel", None)
    if model_cls is None or getattr(model_cls, "_robocerebra_image_feature_patch", False):
        return

    def _embed_image_compat(self, image: torch.Tensor):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        image_outputs = self.paligemma.model.get_image_features(image)
        if hasattr(image_outputs, "pooler_output"):
            features = image_outputs.pooler_output
        else:
            features = image_outputs

        features = features * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    model_cls.embed_image = _embed_image_compat
    model_cls._robocerebra_image_feature_patch = True


def patch_tokenizer_processor_task_ids(metadata: RLDSMetadata) -> None:
    try:
        tokenizer_module = importlib.import_module("lerobot.processor.tokenizer_processor")
        types_module = importlib.import_module("lerobot.types")
    except ModuleNotFoundError:
        return

    tokenizer_step_cls = getattr(tokenizer_module, "TokenizerProcessorStep", None)
    transition_key_cls = getattr(types_module, "TransitionKey", None)
    if tokenizer_step_cls is None or getattr(tokenizer_step_cls, "_robocerebra_task_id_patch", False):
        return

    original_get_task = tokenizer_step_cls.get_task
    index_to_task = metadata.index_to_task

    def _get_task_compat(self, transition):
        complementary_data = None
        if transition_key_cls is not None:
            complementary_data = transition.get(getattr(transition_key_cls, "COMPLEMENTARY_DATA", None))
        if complementary_data is None:
            complementary_data = transition.get("complementary_data", {})
        task = complementary_data.get(self.task_key)
        if isinstance(task, torch.Tensor):
            if task.ndim == 0:
                return [index_to_task.get(int(task.item()), "")]
            return [index_to_task.get(int(idx), "") for idx in task.detach().cpu().tolist()]
        if isinstance(task, list) and task and all(isinstance(item, (int, np.integer)) for item in task):
            return [index_to_task.get(int(idx), "") for idx in task]
        return original_get_task(self, transition)

    tokenizer_step_cls.get_task = _get_task_compat
    tokenizer_step_cls._robocerebra_task_id_patch = True


def build_underlying_argv(args: argparse.Namespace, model_family: str) -> list[str]:
    job_name = args.job_name or f"robocerebra_{model_family}_rlds_finetune"
    output_dir = str(Path(args.output_dir).expanduser().resolve() / job_name)
    pretrained_path = args.pretrained_path or DEFAULT_PRETRAINED_PATHS[model_family]
    policy_repo_id = args.policy_repo_id or default_policy_repo_id(job_name)

    argv = ["lerobot-train"]
    append_override(argv, "dataset.repo_id", args.dataset_repo_id)
    append_override(argv, "dataset.root", str(Path(args.dataset_root).expanduser().resolve()))
    append_override(argv, "dataset.streaming", True)
    append_override(argv, "policy.type", model_family)
    append_override(argv, "policy.repo_id", policy_repo_id)
    append_override(argv, "policy.pretrained_path", pretrained_path)
    append_override(argv, "output_dir", output_dir)
    append_override(argv, "job_name", job_name)
    append_override(argv, "steps", args.steps)
    append_override(argv, "batch_size", args.batch_size)
    append_override(argv, "num_workers", args.num_workers)
    append_override(argv, "seed", args.seed)
    append_override(argv, "policy.device", args.device)
    append_override(argv, "policy.dtype", args.policy_dtype)
    append_override(argv, "policy.compile_model", args.compile_model)
    append_override(argv, "policy.gradient_checkpointing", args.gradient_checkpointing)
    append_override(argv, "policy.freeze_vision_encoder", args.freeze_vision_encoder)
    append_override(argv, "policy.train_expert_only", args.train_expert_only)
    append_override(argv, "wandb.enable", args.wandb)

    normalization_mapping = args.normalization_mapping
    if (
        model_family == "pi05"
        and normalization_mapping is None
        and not args.disable_default_pi05_norm_fallback
    ):
        normalization_mapping = '{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
    append_override(argv, "policy.normalization_mapping", normalization_mapping)

    for extra_arg in args.extra_arg:
        argv.append(extra_arg)

    return argv


def resolve_run_output_dir(args: argparse.Namespace, model_family: str) -> Path:
    job_name = args.job_name or f"robocerebra_{model_family}_rlds_finetune"
    return Path(args.output_dir).expanduser().resolve() / job_name


def resolve_last_checkpoint_dir(checkpoints_dir: Path) -> Path | None:
    last_symlink = checkpoints_dir / "last"
    if last_symlink.exists():
        try:
            resolved = last_symlink.resolve(strict=True)
        except FileNotFoundError:
            resolved = None
        if resolved is not None and resolved.parent == checkpoints_dir:
            return resolved

    numbered_dirs = [path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not numbered_dirs:
        return None
    return max(numbered_dirs, key=lambda path: int(path.name))


def prune_intermediate_checkpoints(output_dir: Path) -> None:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        logging.info("No checkpoints directory found under %s; skipping checkpoint pruning.", output_dir)
        return

    last_checkpoint_dir = resolve_last_checkpoint_dir(checkpoints_dir)
    if last_checkpoint_dir is None:
        logging.warning("Could not determine the last checkpoint under %s; skipping checkpoint pruning.", checkpoints_dir)
        return

    removed = []
    for child in checkpoints_dir.iterdir():
        if child.name == "last" or not child.is_dir():
            continue
        if child.resolve() == last_checkpoint_dir.resolve():
            continue
        shutil.rmtree(child)
        removed.append(child.name)

    if removed:
        logging.info(
            "Deleted %s intermediate checkpoints from %s. Kept only %s.",
            len(removed),
            checkpoints_dir,
            last_checkpoint_dir.name,
        )
    else:
        logging.info("Checkpoint pruning found no intermediate checkpoints to delete under %s.", checkpoints_dir)


def patch_make_dataset(train_module, builder_dirs: list[Path], metadata: RLDSMetadata, max_episodes: int | None):
    def _patched_make_dataset(cfg):
        apply_imagenet_stats_if_requested(cfg, metadata)
        action_delta_indices = getattr(cfg.policy, "action_delta_indices", None)
        if not action_delta_indices:
            raise ValueError("This RLDS bridge currently requires policies with action_delta_indices.")
        action_chunk_size = len(action_delta_indices)
        logging.info(
            "Using direct RLDS streaming dataset with %s episodes and %s frames.",
            metadata.total_episodes,
            metadata.total_frames,
        )
        return RLDSStreamingDataset(
            builder_dirs=builder_dirs,
            metadata=metadata,
            action_chunk_size=action_chunk_size,
            max_episodes=max_episodes,
        )

    train_module.make_dataset = _patched_make_dataset
    try:
        dataset_factory_module = importlib.import_module("lerobot.datasets.factory")
        dataset_factory_module.make_dataset = _patched_make_dataset
    except ModuleNotFoundError:
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    model_family = canonicalize_model_family(args.model_family)
    if model_family not in DEFAULT_PRETRAINED_PATHS:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    disable_tensorflow_gpu()
    builder_dirs = [resolve_builder_dir(path_str) for path_str in args.rlds_dir]
    cache_path = (
        Path(args.metadata_cache).expanduser().resolve()
        if args.metadata_cache is not None
        else compute_default_cache_path(builder_dirs, args.dataset_fps, args.max_episodes)
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 and rank != 0:
        logging.info("Rank %s waiting for rank 0 to prepare RLDS metadata cache: %s", rank, cache_path)
        wait_for_metadata_cache(cache_path)
        metadata = scan_rlds_metadata(
            builder_dirs=builder_dirs,
            fps=args.dataset_fps,
            cache_path=cache_path,
            force_recompute=False,
            max_episodes=args.max_episodes,
        )
    else:
        metadata = scan_rlds_metadata(
            builder_dirs=builder_dirs,
            fps=args.dataset_fps,
            cache_path=cache_path,
            force_recompute=args.force_recompute_metadata,
            max_episodes=args.max_episodes,
        )

    argv = build_underlying_argv(args, model_family)
    print("Resolved direct-RLDS LeRobot training command:")
    print(shlex.join(argv))
    print(f"RLDS builders       : {', '.join(str(path) for path in builder_dirs)}")
    print(f"Metadata cache      : {cache_path}")
    print(f"Scanned episodes    : {metadata.total_episodes}")
    print(f"Scanned frames      : {metadata.total_frames}")

    if args.dry_run:
        return

    train_module = import_lerobot_train_module()
    patch_pi0_image_feature_compat()
    patch_tokenizer_processor_task_ids(metadata)
    patch_make_dataset(train_module, builder_dirs, metadata, args.max_episodes)
    output_dir = resolve_run_output_dir(args, model_family)
    sys.argv = argv
    train_module.main()
    if args.keep_only_last_checkpoint_on_success and rank == 0:
        prune_intermediate_checkpoints(output_dir)


if __name__ == "__main__":
    main()
