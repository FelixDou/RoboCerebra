#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert local RLDS / TFDS exports to a LeRobot dataset.

This path is much faster than replaying raw RoboCerebra demos through LIBERO,
because the RLDS dataset already contains decoded images, state, action, and
language fields in the training convention used by the existing builder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ["NO_GCE_CHECK"] = "true"

try:
    from tensorflow_datasets.core.utils import gcs_utils

    gcs_utils._is_gcs_disabled = True
except ImportError:  # pragma: no cover - optional depending on tfds version
    pass

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional until execution time
    np = None

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional until execution time
    tf = None

try:
    import tensorflow_datasets as tfds
except ImportError:  # pragma: no cover - optional until execution time
    tfds = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from convert_hdf5_to_lerobot import (
    add_frame_compat,
    build_feature_spec,
    create_lerobot_dataset,
    finalize_lerobot_dataset,
    maybe_remove_existing_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one or more local RLDS / TFDS exports into a single local LeRobotDataset."
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
        "--repo_id",
        required=True,
        help="LeRobot dataset repo id or local dataset name, e.g. robocerebra/pi_train.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory where the local LeRobotDataset should be written.",
    )
    parser.add_argument(
        "--robot_type",
        default="panda",
        help="LeRobot robot_type metadata value.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Dataset frame rate metadata. Defaults to 10 to mirror LeRobot LIBERO datasets.",
    )
    parser.add_argument(
        "--image_storage",
        choices=("video", "image"),
        default="video",
        help="Store visual observations as encoded videos or individual images.",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="Video codec passed through when image_storage=video and the installed LeRobot version supports it.",
    )
    parser.add_argument(
        "--batch_encoding_size",
        type=int,
        default=8,
        help="Video batch encoding size passed through when supported by the installed LeRobot version.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap for quick debugging runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing local LeRobot dataset at the target path before exporting.",
    )
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if tf is None:
        missing.append("tensorflow")
    if tfds is None:
        missing.append("tensorflow-datasets")
    if missing:
        raise ImportError(
            "This converter requires the following packages: " + ", ".join(missing)
        )


def disable_tensorflow_gpu() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


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
            grandchild for grandchild in child.iterdir() if grandchild.is_dir() and (grandchild / "dataset_info.json").is_file()
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


def iter_builder_episodes(builder_dir: Path):
    builder = tfds.builder_from_directory(builder_dir=str(builder_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    total_examples = getattr(builder.info.splits["train"], "num_examples", None)
    yield builder, dataset, total_examples


def add_rlds_episode(dataset, episode) -> tuple[int, str]:
    first_task = None
    num_steps = 0

    for step in episode["steps"]:
        observation = step["observation"]
        image = np.asarray(observation["image"].numpy(), dtype=np.uint8)
        wrist_image = np.asarray(observation["wrist_image"].numpy(), dtype=np.uint8)
        state = np.asarray(observation["state"].numpy(), dtype=np.float32)
        action = np.asarray(step["action"].numpy(), dtype=np.float32)

        step_task = decode_text(step["language_instruction"]).strip()
        if first_task is None:
            first_task = step_task

        add_frame_compat(
            dataset,
            {
                "observation.images.image": image,
                "observation.images.wrist_image": wrist_image,
                "observation.state": state,
                "action": action,
            },
            step_task,
        )
        num_steps += 1

    if num_steps == 0:
        return 0, first_task or ""

    dataset.save_episode()
    return num_steps, first_task or ""


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    disable_tensorflow_gpu()

    builder_dirs = [resolve_builder_dir(path_str) for path_str in args.rlds_dir]
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    maybe_remove_existing_dataset(root=root, repo_id=args.repo_id, overwrite=args.overwrite)

    features = build_feature_spec(args.image_storage, args.fps)
    dataset = create_lerobot_dataset(args, features)
    dataset_root = Path(getattr(dataset, "root", root / args.repo_id))

    total_episodes = 0
    total_frames = 0

    try:
        for builder_dir in builder_dirs:
            builder, tf_dataset, total_examples = next(iter_builder_episodes(builder_dir))

            progress = tf_dataset
            if tqdm is not None:
                progress = tqdm(
                    tf_dataset,
                    total=total_examples,
                    desc=f"Converting {builder.info.name}",
                    unit="episode",
                )

            for episode in progress:
                frames_in_episode, _task = add_rlds_episode(dataset, episode)
                if frames_in_episode == 0:
                    continue

                total_episodes += 1
                total_frames += frames_in_episode

                if args.max_episodes is not None and total_episodes >= args.max_episodes:
                    break

            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                break
    finally:
        finalize_lerobot_dataset(dataset)

    print("RLDS -> LeRobot export complete.")
    print(f"Builder dirs : {', '.join(str(path) for path in builder_dirs)}")
    print(f"Dataset root : {dataset_root}")
    print(f"Episodes     : {total_episodes}")
    print(f"Frames       : {total_frames}")
    print(f"FPS          : {args.fps}")
    print(f"Image mode   : {args.image_storage}")


if __name__ == "__main__":
    main()
