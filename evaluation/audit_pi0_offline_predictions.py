#!/usr/bin/env python3
"""Compare PI0 offline predictions against GT actions from a LeRobot dataset.

This diagnostic does not step RoboCerebra. It loads a trained PI0 checkpoint,
samples frames from the local LeRobot dataset used for fine-tuning, runs the
same policy adapter path used during evaluation, and compares the first
predicted action against the stored GT action for each frame.

Use it to localize failures:
  - low offline error but poor rollout: eval stepping / postprocessing issue
  - high offline error: training / normalization / policy input issue
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - local help can run without torch
    torch = None

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


@dataclass(frozen=True)
class DatasetHandle:
    dataset: Any
    repo_id: str
    root: Path


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def infer_repo_id(dataset_root: Path, explicit_repo_id: str | None) -> str:
    if explicit_repo_id:
        return explicit_repo_id
    # Expected local layout: .../lerobot_datasets/robocerebra/<dataset_name>
    if dataset_root.parent.name:
        return f"{dataset_root.parent.name}/{dataset_root.name}"
    return dataset_root.name


def load_lerobot_dataset(dataset_root: Path, repo_id: str) -> DatasetHandle:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = dataset_root.expanduser().resolve()
    candidates = [
        (repo_id, dataset_root),
        (repo_id, dataset_root.parent.parent if dataset_root.parent.parent != dataset_root else dataset_root),
        (dataset_root.name, dataset_root.parent),
    ]
    errors: list[str] = []
    for candidate_repo_id, candidate_root in candidates:
        try:
            dataset = LeRobotDataset(repo_id=candidate_repo_id, root=candidate_root)
            return DatasetHandle(dataset=dataset, repo_id=candidate_repo_id, root=candidate_root)
        except Exception as exc:  # noqa: BLE001 - diagnostics should report all attempted forms
            errors.append(f"repo_id={candidate_repo_id} root={candidate_root}: {exc}")
    raise RuntimeError("Could not instantiate LeRobotDataset:\n" + "\n".join(errors))


def to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = to_numpy(value)
    image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims after squeeze, got {image.shape}")
    if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        # LeRobot datasets often return float images in [0, 1].
        if np.nanmax(image) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def vector_to_float32(value: Any) -> np.ndarray:
    return np.asarray(np.squeeze(to_numpy(value)), dtype=np.float32)


def get_first_available(frame: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in frame:
            return frame[name]
    raise KeyError(f"None of these keys were present: {names}. Available: {sorted(frame)}")


def frame_to_observation(frame: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, str, int, int]:
    full_image = image_to_hwc_uint8(
        get_first_available(frame, ["observation.images.image", "observation.image", "image"])
    )
    try:
        wrist_image = image_to_hwc_uint8(
            get_first_available(frame, ["observation.images.wrist_image", "observation.images.image2", "wrist_image"])
        )
    except KeyError:
        wrist_image = full_image
    state = vector_to_float32(get_first_available(frame, ["observation.state", "observation.states", "state"]))
    action = vector_to_float32(get_first_available(frame, ["action", "actions"]))

    task = frame.get("task", "")
    if isinstance(task, (list, tuple)) and task:
        task = task[0]
    if not isinstance(task, str):
        task = str(task)

    episode_index = int(np.squeeze(to_numpy(frame.get("episode_index", -1))))
    frame_index = int(np.squeeze(to_numpy(frame.get("frame_index", -1))))
    observation = {
        "full_image": full_image,
        "wrist_image": wrist_image,
        "state": state,
    }
    return observation, action, task, episode_index, frame_index


def sample_indices(dataset_len: int, num_samples: int, stride: int | None, seed: int) -> list[int]:
    if stride is not None and stride > 0:
        return list(range(0, dataset_len, stride))[:num_samples]
    rng = np.random.default_rng(seed)
    count = min(num_samples, dataset_len)
    return sorted(int(idx) for idx in rng.choice(dataset_len, size=count, replace=False))


def summarize_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    action_mae = np.asarray([row["action_mae"] for row in rows], dtype=np.float32)
    action_max_abs = np.asarray([row["action_max_abs"] for row in rows], dtype=np.float32)
    gripper_abs = np.asarray([row["gripper_abs_error"] for row in rows], dtype=np.float32)
    pred_norm = np.asarray([row["pred_action_norm"] for row in rows], dtype=np.float32)
    gt_norm = np.asarray([row["gt_action_norm"] for row in rows], dtype=np.float32)
    return {
        "samples": len(rows),
        "action_mae_mean": float(np.mean(action_mae)),
        "action_mae_median": float(np.median(action_mae)),
        "action_mae_max": float(np.max(action_mae)),
        "action_max_abs_mean": float(np.mean(action_max_abs)),
        "action_max_abs_max": float(np.max(action_max_abs)),
        "gripper_abs_error_mean": float(np.mean(gripper_abs)),
        "pred_action_norm_mean": float(np.mean(pred_norm)),
        "gt_action_norm_mean": float(np.mean(gt_norm)),
    }


def audit(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if torch is None:
        raise ImportError("This diagnostic requires torch in the runtime environment.")
    from policy_adapter import initialize_policy, predict_policy_actions, reset_policy_state

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    repo_id = infer_repo_id(dataset_root, args.dataset_repo_id)
    dataset_handle = load_lerobot_dataset(dataset_root, repo_id)
    dataset = dataset_handle.dataset

    cfg = SimpleNamespace(
        model_family="pi0",
        pretrained_checkpoint=str(Path(args.checkpoint).expanduser()),
    )
    policy_runtime = initialize_policy(cfg)

    indices = sample_indices(len(dataset), args.num_samples, args.stride, args.seed)
    rows: list[dict[str, Any]] = []

    for sample_idx, dataset_index in enumerate(indices):
        frame = dataset[dataset_index]
        observation, gt_action, task, episode_index, frame_index = frame_to_observation(frame)
        reset_policy_state(policy_runtime)
        predictions = predict_policy_actions(cfg, policy_runtime, observation, task)
        pred_action = np.asarray(predictions[0], dtype=np.float32)

        if pred_action.shape != gt_action.shape:
            raise ValueError(
                f"Predicted action shape mismatch at dataset index {dataset_index}: "
                f"pred={pred_action.shape}, gt={gt_action.shape}"
            )

        diff = pred_action - gt_action
        rows.append(
            {
                "sample_idx": sample_idx,
                "dataset_index": dataset_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": task,
                "action_mae": float(np.mean(np.abs(diff))),
                "action_max_abs": float(np.max(np.abs(diff))),
                "gripper_abs_error": float(abs(diff[-1])),
                "pred_action_norm": float(np.linalg.norm(pred_action)),
                "gt_action_norm": float(np.linalg.norm(gt_action)),
                "pred_gripper": float(pred_action[-1]),
                "gt_gripper": float(gt_action[-1]),
                "pred_action": json_cell(pred_action.round(6).tolist()),
                "gt_action": json_cell(gt_action.round(6).tolist()),
                "diff": json_cell(diff.round(6).tolist()),
            }
        )

    summary = summarize_errors(rows)
    summary.update(
        {
            "dataset_len": len(dataset),
            "dataset_repo_id_used": dataset_handle.repo_id,
            "dataset_root_used": str(dataset_handle.root),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
        }
    )
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="PI0 checkpoint/pretrained_model path.")
    parser.add_argument("--dataset_root", required=True, help="Local LeRobot dataset directory.")
    parser.add_argument("--dataset_repo_id", default=None, help="Optional dataset repo id, e.g. robocerebra/name.")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of frames to sample.")
    parser.add_argument("--stride", type=int, default=None, help="Optional deterministic frame stride.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for frame sampling.")
    parser.add_argument("--output_csv", required=True, help="Output per-sample CSV path.")
    args = parser.parse_args()

    rows, summary = audit(args)
    write_csv(Path(args.output_csv).expanduser(), rows)
    print("PI0_OFFLINE_PREDICTION_AUDIT_COMPLETE")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  output_csv={Path(args.output_csv).expanduser()}")


if __name__ == "__main__":
    main()
