#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge several local LeRobotDataset shards into one local LeRobotDataset.

This is useful when a long RLDS -> LeRobot export was split into resumable
chunks, but downstream training should see a single standard LeRobot dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional until execution time
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - optional until execution time
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional until execution time
    Image = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from convert_hdf5_to_lerobot import (
    ACTION_SHAPE,
    IMAGE_SHAPE,
    STATE_SHAPE,
    add_frame_compat,
    build_feature_spec,
    create_lerobot_dataset,
    finalize_lerobot_dataset,
    maybe_remove_existing_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge local LeRobotDataset shards into one local LeRobotDataset."
    )
    parser.add_argument(
        "--shard_repo_id",
        action="append",
        required=True,
        help="Shard repo id, e.g. robocerebra/pi_train_lerobot_images_shard_000000. Repeat for each shard.",
    )
    parser.add_argument(
        "--shards_root",
        required=True,
        help="Parent directory containing the local shard directories.",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Output LeRobot dataset repo id, e.g. robocerebra/pi_train_lerobot_images.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory where the merged local LeRobotDataset should be written.",
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
        help="Dataset frame rate metadata.",
    )
    parser.add_argument(
        "--image_storage",
        choices=("video", "image"),
        default="image",
        help="Store visual observations as encoded videos or individual images.",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="Video codec passed through when image_storage=video and supported by LeRobot.",
    )
    parser.add_argument(
        "--batch_encoding_size",
        type=int,
        default=8,
        help="Video batch encoding size passed through when supported by LeRobot.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap for a quick merge smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing merged dataset at the target path before exporting.",
    )
    return parser.parse_args()


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
    raise FileNotFoundError(
        f"Could not resolve local LeRobot dataset root for {repo_id} under {base_root}."
    )


def to_numpy(value):
    if np is None:
        raise ImportError("This merge utility requires numpy.")

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if Image is not None and isinstance(value, Image.Image):
        return np.asarray(value)
    return np.asarray(value)


def scalar_to_int(value) -> int:
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def normalize_image(value) -> np.ndarray:
    image = to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")

    if image.shape == IMAGE_SHAPE:
        pass
    elif image.shape[0] == IMAGE_SHAPE[-1] and image.shape[1:] == IMAGE_SHAPE[:2]:
        image = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"Unexpected image shape {image.shape}; expected {IMAGE_SHAPE} or CHW equivalent.")

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255)

    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def normalize_vector(value, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = to_numpy(value)
    if array.shape != expected_shape:
        raise ValueError(f"Unexpected {name} shape {array.shape}; expected {expected_shape}.")
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def normalize_task(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalize_task(value[0])
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return str(value.detach().cpu().item())
        return " ".join(str(item) for item in value.detach().cpu().tolist())
    return str(value)


def make_output_dataset(args: argparse.Namespace):
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    maybe_remove_existing_dataset(root=root, repo_id=args.repo_id, overwrite=args.overwrite)

    features = build_feature_spec(args.image_storage, args.fps)
    return create_lerobot_dataset(args, features)


def add_sample_frame(target_dataset, sample) -> int:
    task = normalize_task(sample.get("task", ""))
    frame = {
        "observation.images.image": normalize_image(sample["observation.images.image"]),
        "observation.images.wrist_image": normalize_image(sample["observation.images.wrist_image"]),
        "observation.state": normalize_vector(sample["observation.state"], STATE_SHAPE, "state"),
        "action": normalize_vector(sample["action"], ACTION_SHAPE, "action"),
    }
    add_frame_compat(target_dataset, frame, task)
    return 1


def copy_source_dataset(
    target_dataset,
    source_dataset,
    max_episodes: int | None,
    completed_episodes: int,
) -> tuple[int, int]:
    total_episodes = 0
    total_frames = 0
    current_episode_index = None
    pending_frames = 0

    sample_indices = range(len(source_dataset))
    if tqdm is not None:
        sample_indices = tqdm(sample_indices, desc="Merging frames", unit="frame")

    for sample_index in sample_indices:
        sample = source_dataset[sample_index]
        episode_index = scalar_to_int(sample["episode_index"])

        if current_episode_index is None:
            current_episode_index = episode_index
        elif episode_index != current_episode_index:
            if pending_frames > 0:
                target_dataset.save_episode()
                total_episodes += 1
                completed_episodes += 1
                pending_frames = 0
            if max_episodes is not None and completed_episodes >= max_episodes:
                return total_episodes, total_frames
            current_episode_index = episode_index

        total_frames += add_sample_frame(target_dataset, sample)
        pending_frames += 1

    if pending_frames > 0 and (max_episodes is None or completed_episodes < max_episodes):
        target_dataset.save_episode()
        total_episodes += 1

    return total_episodes, total_frames


def main() -> None:
    if np is None:
        raise ImportError("This merge utility requires numpy.")

    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    shards_root = Path(args.shards_root).expanduser().resolve()
    target_dataset = make_output_dataset(args)
    target_root = Path(getattr(target_dataset, "root", Path(args.root).expanduser().resolve() / args.repo_id))

    total_episodes = 0
    total_frames = 0

    try:
        for shard_repo_id in args.shard_repo_id:
            source_root = resolve_local_dataset_root(shards_root, shard_repo_id)
            source_dataset = LeRobotDataset(repo_id=shard_repo_id, root=source_root)
            print(f"Merging {source_root.name}: {source_dataset.num_episodes} episodes, {len(source_dataset)} frames")
            episodes, frames = copy_source_dataset(
                target_dataset=target_dataset,
                source_dataset=source_dataset,
                max_episodes=args.max_episodes,
                completed_episodes=total_episodes,
            )
            total_episodes += episodes
            total_frames += frames

            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                break
    finally:
        finalize_lerobot_dataset(target_dataset)

    print("LeRobot shard merge complete.")
    print(f"Output root : {target_root}")
    print(f"Shards      : {len(args.shard_repo_id)}")
    print(f"Episodes    : {total_episodes}")
    print(f"Frames      : {total_frames}")
    print(f"FPS         : {args.fps}")
    print(f"Image mode  : {args.image_storage}")


if __name__ == "__main__":
    main()
