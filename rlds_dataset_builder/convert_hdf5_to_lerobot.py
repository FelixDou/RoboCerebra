#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export RoboCerebra per-step HDF5 episodes to a local LeRobotDataset.

This converter is intended for pi0 / pi0.5-style training with LeRobot. It
reuses the same image orientation and state construction as the existing RLDS
builder so the LeRobot export stays aligned with the OpenVLA training path.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import h5py
except ImportError:  # pragma: no cover - optional until export time
    h5py = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional until export time
    np = None


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


DEFAULT_FPS = 10
IMAGE_SHAPE = (256, 256, 3)
STATE_SHAPE = (8,)
ACTION_SHAPE = (7,)


@dataclass(frozen=True)
class EpisodeSpec:
    case_name: str
    step_name: str
    instruction: str
    hdf5_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoboCerebra per-step HDF5 episodes into a local LeRobotDataset."
    )
    parser.add_argument(
        "--robocerebra_hdf5_root",
        required=True,
        help="Path to the converted RoboCerebra HDF5 root produced by regenerate_robocerebra_dataset.py.",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="LeRobot dataset repo id or local dataset name, e.g. robocerebra/pi0_train.",
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
        default=DEFAULT_FPS,
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


def resolve_per_step_root(root: Path) -> Path:
    if root.name == "per_step" and root.is_dir():
        return root

    per_step_root = root / "per_step"
    if per_step_root.is_dir():
        return per_step_root

    raise FileNotFoundError(
        f"Could not find a `per_step` directory under {root}. "
        "Run regenerate_robocerebra_dataset.py first and point this script to its output root."
    )


def format_instruction(step_name: str) -> str:
    return step_name.replace("_", " ").strip().rstrip(".")


def iter_episode_specs(per_step_root: Path) -> list[EpisodeSpec]:
    episode_specs: list[EpisodeSpec] = []
    case_dirs = sorted(path for path in per_step_root.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        step_dirs = sorted(path for path in case_dir.iterdir() if path.is_dir())
        for step_dir in step_dirs:
            instruction = format_instruction(step_dir.name)
            for hdf5_path in sorted(step_dir.glob("*.hdf5")):
                episode_specs.append(
                    EpisodeSpec(
                        case_name=case_dir.name,
                        step_name=step_dir.name,
                        instruction=instruction,
                        hdf5_path=hdf5_path,
                    )
                )
    return episode_specs


def build_feature_spec(_image_storage: str, _fps: int) -> dict[str, dict[str, object]]:
    return {
        "observation.images.image": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_image": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": STATE_SHAPE,
            "names": {
                "motors": [
                    "eef_x",
                    "eef_y",
                    "eef_z",
                    "eef_axis_angle_x",
                    "eef_axis_angle_y",
                    "eef_axis_angle_z",
                    "gripper_left",
                    "gripper_right",
                ]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": ACTION_SHAPE,
            "names": {
                "motors": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_axis_angle_x",
                    "delta_axis_angle_y",
                    "delta_axis_angle_z",
                    "gripper",
                ]
            },
        },
    }


def load_episode_arrays(hdf5_path: Path) -> Iterable[dict[str, np.ndarray]]:
    if h5py is None or np is None:
        raise ImportError(
            "This exporter requires both `h5py` and `numpy`. Install them before running the conversion."
        )

    with h5py.File(hdf5_path, "r") as handle:
        data_group = handle["data"]
        for demo_name in sorted(data_group.keys()):
            demo_group = data_group[demo_name]
            obs_group = demo_group["obs"]

            ee_states = np.asarray(obs_group["ee_states"][()], dtype=np.float32)
            gripper_states = np.asarray(obs_group["gripper_states"][()], dtype=np.float32)
            actions = np.asarray(demo_group["actions"][()], dtype=np.float32)
            images = np.asarray(obs_group["agentview_rgb"][()], dtype=np.uint8)
            wrist_images = np.asarray(obs_group["eye_in_hand_rgb"][()], dtype=np.uint8)

            if not len(actions):
                continue

            lengths = {len(ee_states), len(gripper_states), len(actions), len(images), len(wrist_images)}
            if len(lengths) != 1:
                raise ValueError(f"Mismatched sequence lengths in {hdf5_path} ({demo_name}).")

            # Match the existing RLDS builder / evaluation convention for LIBERO frames.
            images = np.ascontiguousarray(images[:, ::-1, ::-1, :])
            wrist_images = np.ascontiguousarray(wrist_images[:, ::-1, ::-1, :])
            state = np.ascontiguousarray(np.concatenate((ee_states, gripper_states), axis=-1), dtype=np.float32)

            yield {
                "demo_name": demo_name,
                "images": images,
                "wrist_images": wrist_images,
                "state": state,
                "actions": np.ascontiguousarray(actions, dtype=np.float32),
            }


def maybe_remove_existing_dataset(root: Path, repo_id: str, overwrite: bool) -> None:
    dataset_dir = root / repo_id
    if not dataset_dir.exists():
        return

    if not overwrite:
        raise FileExistsError(
            f"Target dataset directory already exists: {dataset_dir}\n"
            "Re-run with --overwrite to replace it."
        )

    shutil.rmtree(dataset_dir)


def create_lerobot_dataset(args: argparse.Namespace, features: dict[str, dict[str, object]]):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - depends on external install
        raise ImportError(
            "Could not import LeRobot. Install a LeRobot version with `LeRobotDataset.create` support "
            "before running this exporter."
        ) from exc

    if not hasattr(LeRobotDataset, "create"):
        raise AttributeError(
            "The installed LeRobot version does not expose `LeRobotDataset.create`. "
            "Please upgrade LeRobot to a recent release."
        )

    create_signature = inspect.signature(LeRobotDataset.create)
    create_kwargs = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        "robot_type": args.robot_type,
        "features": features,
        "root": args.root,
    }

    if "use_videos" in create_signature.parameters:
        create_kwargs["use_videos"] = args.image_storage == "video"
    if "vcodec" in create_signature.parameters and args.image_storage == "video":
        create_kwargs["vcodec"] = args.vcodec
    if "batch_encoding_size" in create_signature.parameters and args.image_storage == "video":
        create_kwargs["batch_encoding_size"] = args.batch_encoding_size

    return LeRobotDataset.create(**create_kwargs)


def add_episode(dataset, episode_spec: EpisodeSpec, episode_arrays: dict[str, np.ndarray]) -> int:
    task = episode_spec.instruction
    images = episode_arrays["images"]
    wrist_images = episode_arrays["wrist_images"]
    state = episode_arrays["state"]
    actions = episode_arrays["actions"]

    for frame_idx in range(len(actions)):
        frame = {
            "observation.images.image": images[frame_idx],
            "observation.images.wrist_image": wrist_images[frame_idx],
            "observation.state": state[frame_idx],
            "action": actions[frame_idx],
        }
        dataset.add_frame(frame, task=task)

    dataset.save_episode()
    return len(actions)


def finalize_lerobot_dataset(dataset) -> None:
    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
        return

    consolidate = getattr(dataset, "consolidate", None)
    if callable(consolidate):  # pragma: no cover - compatibility fallback
        consolidate()
        return

    raise AttributeError("The created dataset exposes neither `finalize()` nor `consolidate()`.")


def main() -> None:
    args = parse_args()
    if np is None:
        raise ImportError("This exporter requires `numpy`. Install it before running the conversion.")

    input_root = Path(args.robocerebra_hdf5_root).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    per_step_root = resolve_per_step_root(input_root)
    episode_specs = iter_episode_specs(per_step_root)

    if args.max_episodes is not None:
        episode_specs = episode_specs[: args.max_episodes]

    if not episode_specs:
        raise RuntimeError(f"No per-step HDF5 episodes found under {per_step_root}.")

    root.mkdir(parents=True, exist_ok=True)
    maybe_remove_existing_dataset(root=root, repo_id=args.repo_id, overwrite=args.overwrite)

    features = build_feature_spec(args.image_storage, args.fps)
    dataset = create_lerobot_dataset(args, features)

    total_frames = 0
    total_episodes = 0
    progress_iter = episode_specs
    if tqdm is not None:
        progress_iter = tqdm(episode_specs, desc="Exporting LeRobot episodes", unit="episode")

    try:
        for episode_spec in progress_iter:
            for episode_arrays in load_episode_arrays(episode_spec.hdf5_path):
                total_frames += add_episode(dataset, episode_spec, episode_arrays)
                total_episodes += 1
    finally:
        finalize_lerobot_dataset(dataset)

    print("LeRobot export complete.")
    print(f"Input HDF5 root : {per_step_root}")
    print(f"Dataset root    : {root / args.repo_id}")
    print(f"Episodes        : {total_episodes}")
    print(f"Frames          : {total_frames}")
    print(f"FPS             : {args.fps}")
    print(f"Image storage   : {args.image_storage}")


if __name__ == "__main__":
    main()
