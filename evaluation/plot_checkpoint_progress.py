#!/usr/bin/env python3
"""Create checkpoint-evolution charts from RoboCerebra evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape


TASK_ORDER = [
    "Ideal",
    "Memory_Execution",
    "Memory_Exploration",
    "Mix",
    "Observation_Mismatching",
    "Random_Disturbance",
]

TASK_COLORS = {
    "Ideal": "#4C78A8",
    "Memory_Execution": "#E45756",
    "Memory_Exploration": "#72B7B2",
    "Mix": "#F58518",
    "Observation_Mismatching": "#54A24B",
    "Random_Disturbance": "#B279A2",
}

DISPLAY_LABELS = {
    "Ideal": "Ideal",
    "Memory_Execution": "Memory exec.",
    "Memory_Exploration": "Memory expl.",
    "Mix": "Mix",
    "Observation_Mismatching": "Obs. mismatch",
    "Random_Disturbance": "Random disturb.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Step-0 evaluation JSON.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        type=Path,
        help="Checkpoint evaluation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("assets/figures"),
        type=Path,
        help="Directory where charts and summary CSV will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="pi05_checkpoint_progress",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--model-label",
        default="pi0.5",
        help="Human-readable model label used in chart titles.",
    )
    return parser.parse_args()


def display_label(task_name: str) -> str:
    return DISPLAY_LABELS.get(task_name, task_name.replace("_", " "))


def step_label(step: int) -> str:
    if step == 0:
        return "0"
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def parse_step(path: Path, fallback: int = 0) -> int:
    match = re.search(r"ckpt_(\d+)", path.name)
    return int(match.group(1)) if match else fallback


def read_eval(path: Path, step: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_task = data["results_by_task_type"]
    return {
        "step": step,
        "run_id": data["evaluation_info"].get("run_id", path.stem),
        "model_family": data["evaluation_info"].get("model_family", "model"),
        "num_trials_per_task": data["evaluation_info"].get("num_trials_per_task", ""),
        "total_episodes": data["overall_results"]["total_episodes"],
        "overall_subtask_rate": data["overall_results"]["overall_subtask_rate"] * 100.0,
        "overall_success_rate": data["overall_results"].get("overall_success_rate", 0.0) * 100.0,
        "task_rates": {
            task: by_task[task]["subtask_rate"] * 100.0
            for task in TASK_ORDER
            if task in by_task
        },
        "task_counts": {
            task: (by_task[task]["agent_subtasks"], by_task[task]["possible_subtasks"])
            for task in TASK_ORDER
            if task in by_task
        },
        "path": str(path),
    }


def load_runs(baseline: Path, checkpoints: list[Path]) -> list[dict]:
    runs = [read_eval(baseline, 0)]
    runs.extend(read_eval(path, parse_step(path)) for path in checkpoints)
    return sorted(runs, key=lambda run: run["step"])


def nice_y_max(values: list[float], minimum: float = 25.0) -> float:
    max_value = max(values) if values else minimum
    return max(minimum, math.ceil((max_value + 2.0) / 5.0) * 5.0)


def color_for_value(value: float, vmax: float) -> str:
    vmax = max(vmax, 1e-6)
    ratio = max(0.0, min(1.0, value / vmax))
    start = (255, 247, 188)
    end = (8, 88, 158)
    rgb = tuple(round(start[i] + ratio * (end[i] - start[i])) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def write_csv(runs: list[dict], output_path: Path) -> None:
    tasks = [task for task in TASK_ORDER if any(task in run["task_rates"] for run in runs)]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "overall_subtask_rate_percent",
                "overall_success_rate_percent",
                "total_episodes",
                "num_trials_per_task",
                *[f"{task}_subtask_rate_percent" for task in tasks],
                "run_id",
                "path",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run["step"],
                    f'{run["overall_subtask_rate"]:.6f}',
                    f'{run["overall_success_rate"]:.6f}',
                    run["total_episodes"],
                    run["num_trials_per_task"],
                    *[f'{run["task_rates"].get(task, math.nan):.6f}' for task in tasks],
                    run["run_id"],
                    run["path"],
                ]
            )


def save_overall_svg(runs: list[dict], output_path: Path, model_label: str) -> None:
    width = 1600
    height = 940
    margin_l = 140
    margin_r = 90
    margin_t = 175
    margin_b = 175
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x_min = min(run["step"] for run in runs)
    x_max = max(run["step"] for run in runs)
    y_max = nice_y_max([run["overall_subtask_rate"] for run in runs])

    def x_pos(step: int) -> float:
        return margin_l + ((step - x_min) / max(1, x_max - x_min)) * plot_w

    def y_pos(value: float) -> float:
        return margin_t + (1.0 - value / y_max) * plot_h

    best = max(runs, key=lambda run: run["overall_subtask_rate"])
    points = [(x_pos(run["step"]), y_pos(run["overall_subtask_rate"])) for run in runs]
    point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf5"/>',
        f'<text x="800" y="64" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="700" fill="#151515">{escape(model_label)} fine-tuning checkpoint evolution</text>',
        '<text x="800" y="102" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#474747">Overall subtask completion on RoboCerebra</text>',
        f'<text x="800" y="130" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#666">Step 0 uses the previous {escape(model_label)} evaluation as the baseline</text>',
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" rx="16" fill="#ffffff" stroke="#ddd8ca"/>',
    ]

    for tick in range(0, int(y_max) + 1, 5):
        y = y_pos(tick)
        lines.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}" stroke="#e8e4d8" stroke-dasharray="5 6"/>')
        lines.append(
            f'<text x="{margin_l - 18}" y="{y + 6:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">{tick}</text>'
        )

    for run in runs:
        x = x_pos(run["step"])
        lines.append(f'<line x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{margin_t + plot_h}" stroke="#efeadc" stroke-width="1"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{margin_t + plot_h + 30}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#555">{step_label(run["step"])}</text>'
        )

    lines.extend(
        [
            f'<polyline points="{point_string}" fill="none" stroke="#2B6CB0" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>',
        ]
    )

    for run, (x, y) in zip(runs, points):
        fill = "#F58518" if run is best else "#ffffff"
        stroke = "#F58518" if run is best else "#2B6CB0"
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{fill}" stroke="{stroke}" stroke-width="4"/>')
        if run is best:
            lines.append(
                f'<text x="{x + 16:.1f}" y="{y - 16:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700" fill="#1f1f1f">Best: {run["overall_subtask_rate"]:.1f}% at {step_label(run["step"])}</text>'
            )

    lines.append(
        f'<text x="{margin_l + plot_w / 2:.1f}" y="{height - 72}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="#333">Training checkpoint step</text>'
    )
    lines.append(
        f'<text x="52" y="{margin_t + plot_h / 2:.1f}" transform="rotate(-90 52,{margin_t + plot_h / 2:.1f})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="#333">Subtask completion (%)</text>'
    )
    lines.append(
        f'<text x="{margin_l}" y="{height - 28}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#666">Note: checkpoint evaluations use {runs[-1]["num_trials_per_task"]} trial per task; the step-0 baseline uses {runs[0]["num_trials_per_task"]} trials per task.</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_category_svg(runs: list[dict], output_path: Path, model_label: str) -> None:
    width = 1600
    height = 940
    margin_l = 140
    margin_r = 315
    margin_t = 175
    margin_b = 175
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x_min = min(run["step"] for run in runs)
    x_max = max(run["step"] for run in runs)
    tasks = [task for task in TASK_ORDER if any(task in run["task_rates"] for run in runs)]
    y_values = [run["task_rates"][task] for run in runs for task in tasks if task in run["task_rates"]]
    y_max = nice_y_max(y_values)

    def x_pos(step: int) -> float:
        return margin_l + ((step - x_min) / max(1, x_max - x_min)) * plot_w

    def y_pos(value: float) -> float:
        return margin_t + (1.0 - value / y_max) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf5"/>',
        f'<text x="800" y="64" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="700" fill="#151515">{escape(model_label)} checkpoint evolution by task type</text>',
        '<text x="800" y="102" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#474747">Subtask completion across RoboCerebra evaluation categories</text>',
        f'<text x="800" y="130" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#666">Step 0 uses the previous {escape(model_label)} evaluation as the baseline</text>',
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" rx="16" fill="#ffffff" stroke="#ddd8ca"/>',
    ]

    for tick in range(0, int(y_max) + 1, 5):
        y = y_pos(tick)
        lines.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}" stroke="#e8e4d8" stroke-dasharray="5 6"/>')
        lines.append(
            f'<text x="{margin_l - 18}" y="{y + 6:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">{tick}</text>'
        )

    for run in runs:
        x = x_pos(run["step"])
        lines.append(f'<line x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{margin_t + plot_h}" stroke="#efeadc" stroke-width="1"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{margin_t + plot_h + 30}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#555">{step_label(run["step"])}</text>'
        )

    for task in tasks:
        task_points = [
            (x_pos(run["step"]), y_pos(run["task_rates"][task]))
            for run in runs
            if task in run["task_rates"]
        ]
        point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in task_points)
        color = TASK_COLORS.get(task, "#777777")
        lines.append(
            f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" opacity="0.95"/>'
        )
        for x, y in task_points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#ffffff" stroke="{color}" stroke-width="3"/>')

    legend_x = margin_l + plot_w + 42
    legend_y = margin_t + 16
    for idx, task in enumerate(tasks):
        y = legend_y + idx * 42
        color = TASK_COLORS.get(task, "#777777")
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{legend_x + 17}" cy="{y}" r="5" fill="#ffffff" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x + 48}" y="{y + 6}" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#222">{escape(display_label(task))}</text>'
        )

    lines.append(
        f'<text x="{margin_l + plot_w / 2:.1f}" y="{height - 72}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="#333">Training checkpoint step</text>'
    )
    lines.append(
        f'<text x="52" y="{margin_t + plot_h / 2:.1f}" transform="rotate(-90 52,{margin_t + plot_h / 2:.1f})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="#333">Subtask completion (%)</text>'
    )
    lines.append(
        f'<text x="{margin_l}" y="{height - 28}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#666">Note: category curves are noisy because checkpoint runs use fewer trials than the step-0 baseline.</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_heatmap_svg(runs: list[dict], output_path: Path, model_label: str) -> None:
    width = 1800
    height = 940
    left = 120
    right = 145
    top = 180
    bottom = 155
    panel_x = left
    panel_y = top
    panel_w = width - left - right
    panel_h = height - top - bottom
    label_w = 190
    legend_w = 20
    legend_gap = 24
    heat_x = panel_x + label_w
    heat_y = panel_y + 32
    heat_w = panel_w - label_w - legend_w - legend_gap - 40
    heat_h = panel_h - 80
    tasks = [task for task in TASK_ORDER if any(task in run["task_rates"] for run in runs)]
    max_rate = nice_y_max([run["task_rates"][task] for run in runs for task in tasks if task in run["task_rates"]])
    cell_w = heat_w / len(runs)
    cell_h = heat_h / len(tasks)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf5"/>',
        f'<text x="900" y="64" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="700" fill="#151515">{escape(model_label)} checkpoint heatmap</text>',
        '<text x="900" y="102" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#474747">Subtask completion by task type and training step</text>',
        f'<text x="900" y="130" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#666">Step 0 uses the previous {escape(model_label)} evaluation as the baseline</text>',
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="18" fill="#ffffff" stroke="#ddd8ca"/>',
    ]

    for j, run in enumerate(runs):
        x = heat_x + j * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x:.1f}" y="{heat_y + heat_h + 34:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#555">{step_label(run["step"])}</text>'
        )

    for i, task in enumerate(tasks):
        y = heat_y + i * cell_h + cell_h / 2 + 8
        lines.append(
            f'<text x="{panel_x + 24}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#1f1f1f">{escape(display_label(task))}</text>'
        )
        for j, run in enumerate(runs):
            value = run["task_rates"].get(task, math.nan)
            x = heat_x + j * cell_w
            y0 = heat_y + i * cell_h
            if math.isnan(value):
                fill = "#eeeeee"
                label = "NA"
                text_color = "#666"
            else:
                fill = color_for_value(value, max_rate)
                label = f"{value:.1f}"
                text_color = "#ffffff" if value >= max_rate * 0.55 else "#10212b"
            lines.append(
                f'<rect x="{x:.1f}" y="{y0:.1f}" width="{cell_w - 4:.1f}" height="{cell_h - 6:.1f}" rx="7" fill="{fill}" stroke="#ffffff" stroke-width="1.5"/>'
            )
            lines.append(
                f'<text x="{x + (cell_w - 4) / 2:.1f}" y="{y0 + (cell_h - 6) / 2 + 7:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" font-weight="700" fill="{text_color}">{label}</text>'
            )

    legend_x = heat_x + heat_w + legend_gap
    legend_y = heat_y
    legend_h = heat_h
    for step in range(140):
        ratio = step / 140
        y = legend_y + (1 - ratio) * legend_h
        color = color_for_value(ratio * max_rate, max_rate)
        lines.append(f'<rect x="{legend_x:.1f}" y="{y:.1f}" width="{legend_w}" height="{legend_h / 140 + 1:.2f}" fill="{color}" stroke="none"/>')
    lines.append(
        f'<text x="{legend_x + legend_w + 10:.1f}" y="{legend_y + 8:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">{max_rate:.0f}</text>'
    )
    lines.append(
        f'<text x="{legend_x + legend_w + 10:.1f}" y="{legend_y + legend_h:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">0</text>'
    )
    lines.append(
        f'<text x="{legend_x + 56:.1f}" y="{legend_y + legend_h / 2:.1f}" transform="rotate(90 {legend_x + 56:.1f},{legend_y + legend_h / 2:.1f})" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">Completion (%)</text>'
    )

    lines.append(
        f'<text x="{panel_x}" y="{height - 36}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#666">Checkpoint runs use fewer trials than the step-0 baseline; read this as a trend and sanity check rather than a high-confidence ranking.</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = load_runs(args.baseline, args.checkpoints)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(runs, args.output_dir / f"{args.prefix}_summary.csv")
    save_overall_svg(runs, args.output_dir / f"{args.prefix}_overall.svg", args.model_label)
    save_category_svg(runs, args.output_dir / f"{args.prefix}_by_category.svg", args.model_label)
    save_heatmap_svg(runs, args.output_dir / f"{args.prefix}_heatmap.svg", args.model_label)

    print(f"Wrote {len(runs)} checkpoint summaries")
    print(f"Best overall subtask completion: {max(runs, key=lambda run: run['overall_subtask_rate'])['overall_subtask_rate']:.2f}%")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
