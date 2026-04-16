#!/usr/bin/env python3
"""Inspect temporal action-window consistency in local LeRobot shards."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.finetune_lerobot_policy import parse_dataset_repo_ids, resolve_local_dataset_root


ACTION_COLUMN_CANDIDATES = ("action", "actions")
EPISODE_COLUMN_CANDIDATES = ("episode_index", "episode_idx", "episode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample consecutive action windows from local LeRobot parquet files and report "
            "how much future actions differ from the first action in each window."
        )
    )
    parser.add_argument("--dataset_repo_id", required=True, help="Dataset repo id, comma list, or JSON list.")
    parser.add_argument("--dataset_root", required=True, help="Root containing local LeRobot dataset shards.")
    parser.add_argument("--window_size", type=int, default=3, help="Consecutive action window size to inspect.")
    parser.add_argument("--samples", type=int, default=5000, help="Maximum valid windows to sample.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed.")
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on parquet files to scan, useful for quick smoke checks.",
    )
    return parser.parse_args()


def discover_parquet_files(repo_ids: list[str], dataset_root: Path) -> list[Path]:
    parquet_files = []
    for repo_id in repo_ids:
        shard_root = resolve_local_dataset_root(dataset_root, repo_id)
        data_root = shard_root / "data"
        if not data_root.is_dir():
            raise FileNotFoundError(f"Could not find LeRobot data directory: {data_root}")
        parquet_files.extend(sorted(data_root.rglob("*.parquet")))

    if not parquet_files:
        raise FileNotFoundError("No parquet files found under the requested LeRobot shard data directories.")
    return parquet_files


def import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for this inspector. Install pyarrow in the training environment, "
            "or run from the robocerebra-pi0 environment if it already includes it."
        ) from exc
    return pq


def choose_column(columns: list[str], candidates: tuple[str, ...], label: str) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if label == "action":
        raise KeyError(f"Could not find an action column. Available columns include: {columns[:20]}")
    return None


def as_1d_float_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def sample_file_windows(table, action_column: str, episode_column: str | None, window_size: int, limit: int, rng):
    actions = table[action_column].to_pylist()
    episodes = table[episode_column].to_pylist() if episode_column is not None else None
    if len(actions) < window_size:
        return []

    valid_starts = []
    for start in range(0, len(actions) - window_size + 1):
        if episodes is not None:
            episode = episodes[start]
            if any(episodes[start + offset] != episode for offset in range(1, window_size)):
                continue
        valid_starts.append(start)

    if not valid_starts:
        return []
    if len(valid_starts) > limit:
        valid_starts = rng.sample(valid_starts, limit)

    windows = []
    for start in valid_starts:
        window = np.stack([as_1d_float_array(actions[start + offset]) for offset in range(window_size)], axis=0)
        windows.append(window)
    return windows


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def summarize_offset(deltas: np.ndarray, baseline: np.ndarray, offset: int) -> None:
    raw_l2 = np.linalg.norm(deltas, axis=1)
    raw_max_abs = np.max(np.abs(deltas), axis=1)
    sample_std = np.std(baseline, axis=0)
    normalized_rms = np.sqrt(np.mean((deltas / np.maximum(sample_std, 1e-8)) ** 2, axis=1))

    near_same = float(np.mean(normalized_rms < 0.05) * 100.0)
    moderate_or_more = float(np.mean(normalized_rms >= 0.25) * 100.0)
    large = float(np.mean(normalized_rms >= 0.50) * 100.0)

    print(f"\nOffset +{offset} vs first action")
    print(f"  raw_l2:         mean={raw_l2.mean():.6g} p50={percentile(raw_l2, 50):.6g} p95={percentile(raw_l2, 95):.6g}")
    print(
        f"  raw_max_abs:    mean={raw_max_abs.mean():.6g} "
        f"p50={percentile(raw_max_abs, 50):.6g} p95={percentile(raw_max_abs, 95):.6g}"
    )
    print(
        f"  sample-z rms:   mean={normalized_rms.mean():.6g} "
        f"p50={percentile(normalized_rms, 50):.6g} p95={percentile(normalized_rms, 95):.6g}"
    )
    print(f"  sample-z bands: <0.05={near_same:.1f}% >=0.25={moderate_or_more:.1f}% >=0.50={large:.1f}%")


def print_interpretation(windows: np.ndarray) -> None:
    baseline = windows[:, 0, :]
    worst_offset_rms = []
    sample_std = np.std(baseline, axis=0)
    for offset in range(1, windows.shape[1]):
        deltas = windows[:, offset, :] - baseline
        normalized_rms = np.sqrt(np.mean((deltas / np.maximum(sample_std, 1e-8)) ** 2, axis=1))
        worst_offset_rms.append(float(np.percentile(normalized_rms, 95)))

    worst_p95 = max(worst_offset_rms) if worst_offset_rms else 0.0
    print("\nInterpretation")
    if worst_p95 < 0.10:
        print("  The short windows are very consistent. Broadcast supervision is unlikely to matter much.")
    elif worst_p95 < 0.30:
        print("  The short windows move a little. Broadcast supervision is probably acceptable for smoke tests.")
    else:
        print(
            "  The short windows vary meaningfully. For final training quality, prefer fixing action-window "
            "construction instead of relying on loss broadcasting."
        )


def main() -> None:
    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window_size must be at least 2.")
    if args.samples < 1:
        raise ValueError("--samples must be positive.")

    rng = random.Random(args.seed)
    repo_ids = parse_dataset_repo_ids(args.dataset_repo_id)
    parquet_files = discover_parquet_files(repo_ids, Path(args.dataset_root))
    rng.shuffle(parquet_files)
    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]

    pq = import_pyarrow_parquet()

    windows = []
    scanned_files = 0
    for file_index, parquet_path in enumerate(parquet_files):
        remaining_files = len(parquet_files) - file_index
        remaining_samples = args.samples - len(windows)
        if remaining_samples <= 0:
            break

        per_file_limit = max(1, math.ceil(remaining_samples / max(remaining_files, 1)))
        schema = pq.read_schema(parquet_path)
        columns = list(schema.names)
        action_column = choose_column(columns, ACTION_COLUMN_CANDIDATES, "action")
        episode_column = choose_column(columns, EPISODE_COLUMN_CANDIDATES, "episode")
        read_columns = [action_column]
        if episode_column is not None:
            read_columns.append(episode_column)

        table = pq.read_table(parquet_path, columns=read_columns)
        windows.extend(sample_file_windows(table, action_column, episode_column, args.window_size, per_file_limit, rng))
        scanned_files += 1

    if not windows:
        raise RuntimeError("No valid same-episode action windows were sampled.")

    windows_array = np.stack(windows[: args.samples], axis=0)
    baseline = windows_array[:, 0, :]
    print(f"Repo ids: {len(repo_ids)}")
    print(f"Parquet files scanned: {scanned_files} / {len(parquet_files)}")
    print(f"Valid windows sampled: {len(windows_array)}")
    print(f"Window size: {args.window_size}")
    print(f"Action dim: {windows_array.shape[-1]}")

    for offset in range(1, args.window_size):
        summarize_offset(windows_array[:, offset, :] - baseline, baseline, offset)
    print_interpretation(windows_array)


if __name__ == "__main__":
    main()
