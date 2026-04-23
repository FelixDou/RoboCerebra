#!/usr/bin/env python3
"""Create presentation-friendly charts from RoboCerebra evaluation JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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
    parser.add_argument("json_path", type=Path, help="Path to evaluation results JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/figures/openvla_subtask_results.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--variant",
        choices=["full", "slide", "slide_heatmap"],
        default="full",
        help="Chart style to generate",
    )
    return parser.parse_args()


def load_results(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_case_id(case_name: str) -> int:
    digits = "".join(ch for ch in case_name if ch.isdigit())
    return int(digits) if digits else 0


def build_heatmap(task_results: list[dict]) -> tuple[list[list[float]], list[str], list[str]]:
    task_types = [t for t in TASK_ORDER if any(r["task_type"] == t for r in task_results)]
    case_names = sorted(
        {r["case_name"] for r in task_results},
        key=lambda name: (extract_case_id(name), name),
    )
    matrix = [[math.nan for _ in case_names] for _ in task_types]

    for i, task_type in enumerate(task_types):
        for j, case_name in enumerate(case_names):
            match = next(
                (
                    r["subtask_rate"]
                    for r in task_results
                    if r["task_type"] == task_type and r["case_name"] == case_name
                ),
                math.nan,
            )
            matrix[i][j] = float(match) * 100 if not math.isnan(match) else math.nan

    return matrix, task_types, case_names


def short_case_label(case_name: str) -> str:
    case_id = extract_case_id(case_name)
    return f"C{case_id}" if case_id else case_name


def display_label(task_name: str) -> str:
    return DISPLAY_LABELS.get(task_name, task_name.replace("_", " "))


def color_for_value(value: float, vmax: float) -> str:
    """Map 0..vmax to a light-to-dark blue-green color."""
    vmax = max(vmax, 1e-6)
    ratio = max(0.0, min(1.0, value / vmax))
    start = (255, 247, 188)
    end = (8, 88, 158)
    rgb = tuple(round(start[i] + ratio * (end[i] - start[i])) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def max_finite(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return max(finite) if finite else 0.0


def flatten(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def save_svg_chart(
    output_path: Path,
    run_id: str,
    ordered_tasks: list[str],
    subtask_rates: list[float],
    numerator_labels: list[str],
    heatmap: list[list[float]],
    heatmap_tasks: list[str],
    heatmap_cases: list[str],
) -> None:
    width = 1600
    height = 900
    margin = 60
    gutter = 44

    left_x = margin
    left_y = 182
    left_w = 720
    left_h = 560

    right_x = left_x + left_w + gutter
    right_y = left_y
    right_w = 716
    right_h = 560

    max_rate = max(35.0, max(subtask_rates) + 5.0, max_finite(flatten(heatmap)))
    task_labels = [display_label(task) for task in ordered_tasks]
    heatmap_task_labels = [display_label(task) for task in heatmap_tasks]
    case_labels = [short_case_label(case_name) for case_name in heatmap_cases]

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcf8"/>',
        '<text x="800" y="52" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="31" font-weight="700" fill="#1b1b1b">OpenVLA on RoboCerebra</text>',
        '<text x="800" y="84" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#444">LIBERO Spatial benchmark | Subtask completion view</text>',
        f'<text x="800" y="108" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#666">Run: {escape(run_id)}</text>',
        f'<text x="{left_x}" y="{left_y - 34}" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#1b1b1b">Aggregate by task type</text>',
        f'<text x="{right_x}" y="{right_y - 34}" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#1b1b1b">Per-case completion heatmap</text>',
    ]

    # Left panel axes and grid.
    lines.append(f'<rect x="{left_x}" y="{left_y}" width="{left_w}" height="{left_h}" fill="#ffffff" stroke="#d8d8cf"/>')
    for tick in range(0, int(max_rate) + 1, 5):
        x = left_x + (tick / max_rate) * (left_w - 220) + 150
        lines.append(f'<line x1="{x:.1f}" y1="{left_y}" x2="{x:.1f}" y2="{left_y + left_h}" stroke="#e3e3dc" stroke-dasharray="4 4"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{left_y + left_h + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555">{tick}</text>'
        )
    lines.append(
        f'<text x="{left_x + left_w / 2:.1f}" y="{left_y + left_h + 48}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#333">Subtask completion rate (%)</text>'
    )

    bar_area_x = left_x + 150
    bar_area_w = left_w - 210
    row_h = left_h / len(ordered_tasks)
    bar_h = row_h * 0.52

    for idx, (task, task_label) in enumerate(zip(ordered_tasks, task_labels)):
        rate = subtask_rates[idx]
        counts = numerator_labels[idx]
        bar_y = left_y + idx * row_h + (row_h - bar_h) / 2
        bar_w = (rate / max_rate) * bar_area_w
        cy = bar_y + bar_h / 2
        color = TASK_COLORS.get(task, "#777777")
        lines.append(
            f'<text x="{left_x + 16}" y="{cy + 5:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#222">{escape(task_label)}</text>'
        )
        lines.append(
            f'<rect x="{bar_area_x}" y="{bar_y:.1f}" width="{bar_area_w:.1f}" height="{bar_h:.1f}" rx="4" fill="#f0f0ea"/>'
        )
        lines.append(
            f'<rect x="{bar_area_x}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{bar_area_x + bar_w + 8:.1f}" y="{cy + 5:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#222">{rate:.1f}% ({escape(counts)})</text>'
        )

    # Right panel heatmap.
    lines.append(f'<rect x="{right_x}" y="{right_y}" width="{right_w}" height="{right_h}" fill="#ffffff" stroke="#d8d8cf"/>')
    cell_w = (right_w - 170) / len(heatmap_cases)
    cell_h = (right_h - 58) / len(heatmap_tasks)
    heat_x = right_x + 130
    heat_y = right_y + 18

    for j, case_name in enumerate(case_labels):
        tx = heat_x + j * cell_w + cell_w / 2
        lines.append(
            f'<text x="{tx:.1f}" y="{right_y + right_h + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555">{escape(case_name)}</text>'
        )

    for i, task_name in enumerate(heatmap_task_labels):
        ty = heat_y + i * cell_h + cell_h / 2 + 5
        lines.append(
            f'<text x="{right_x + 14}" y="{ty:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#222">{escape(task_name)}</text>'
        )
        for j, _ in enumerate(heatmap_cases):
            value = float(heatmap[i][j])
            x = heat_x + j * cell_w
            y = heat_y + i * cell_h
            fill = color_for_value(value, max_rate)
            text_color = "#ffffff" if value >= 18 else "#10212b"
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 3:.1f}" height="{cell_h - 3:.1f}" rx="3" fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{x + cell_w / 2 - 1:.1f}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="{text_color}">{value:.1f}</text>'
            )

    legend_x = right_x + right_w - 26
    legend_y = heat_y
    legend_h = cell_h * len(heatmap_tasks)
    for step in range(100):
        ratio0 = step / 100
        y = legend_y + (1 - ratio0) * legend_h
        color = color_for_value(ratio0 * max_rate, max_rate)
        lines.append(
            f'<rect x="{legend_x:.1f}" y="{y:.1f}" width="16" height="{legend_h / 100 + 1:.2f}" fill="{color}" stroke="none"/>'
        )
    lines.append(
        f'<text x="{legend_x + 24:.1f}" y="{legend_y + 4}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#444">{max_rate:.0f}</text>'
    )
    lines.append(
        f'<text x="{legend_x + 24:.1f}" y="{legend_y + legend_h}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#444">0</text>'
    )
    lines.append(
        f'<text x="{legend_x + 48:.1f}" y="{legend_y + legend_h / 2:.1f}" transform="rotate(90 {legend_x + 48:.1f},{legend_y + legend_h / 2:.1f})" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#444">Completion (%)</text>'
    )

    lines.append(
        '<text x="60" y="804" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="700" fill="#1b1b1b">Takeaway</text>'
    )
    lines.append(
        '<text x="60" y="832" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#333">Episode success is omitted because no rollout fully satisfied all goal conditions.</text>'
    )
    lines.append(
        '<text x="60" y="854" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#333">This view focuses on partial progress: completed subtasks divided by possible subtasks.</text>'
    )
    lines.append("</svg>")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_slide_svg_chart(
    output_path: Path,
    run_id: str,
    ordered_tasks: list[str],
    subtask_rates: list[float],
    numerator_labels: list[str],
) -> None:
    width = 1600
    height = 940
    left = 110
    right = 90
    top = 170
    bottom = 170
    chart_x = 360
    chart_y = top
    chart_w = width - chart_x - right
    chart_h = height - chart_y - bottom
    max_rate = max(35.0, max(subtask_rates) + 5.0)

    task_labels = [display_label(task) for task in ordered_tasks]
    bars = list(zip(task_labels, ordered_tasks, subtask_rates, numerator_labels))
    bars.sort(key=lambda item: item[2], reverse=True)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf5"/>',
        '<text x="800" y="64" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="700" fill="#151515">OpenVLA subtask completion on RoboCerebra</text>',
        '<text x="800" y="102" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#474747">LIBERO Spatial | Aggregate by evaluation setting</text>',
        f'<text x="800" y="128" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#666">Run: {escape(run_id)}</text>',
        f'<rect x="{left}" y="{chart_y}" width="{width - left - right}" height="{chart_h}" rx="18" fill="#ffffff" stroke="#ddd8ca"/>',
    ]

    for tick in range(0, int(max_rate) + 1, 5):
        x = chart_x + (tick / max_rate) * chart_w
        lines.append(f'<line x1="{x:.1f}" y1="{chart_y}" x2="{x:.1f}" y2="{chart_y + chart_h}" stroke="#e8e4d8" stroke-dasharray="5 6"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{chart_y + chart_h + 32}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#666">{tick}</text>'
        )
    lines.append(
        f'<text x="{chart_x + chart_w / 2:.1f}" y="{chart_y + chart_h + 58}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#333">Completed subtasks / possible subtasks (%)</text>'
    )

    row_h = chart_h / len(bars)
    bar_h = row_h * 0.56

    for idx, (task_label, task_key, rate, counts) in enumerate(bars):
        bar_y = chart_y + idx * row_h + (row_h - bar_h) / 2
        cy = bar_y + bar_h / 2
        bar_w = (rate / max_rate) * chart_w
        color = TASK_COLORS.get(task_key, "#777777")

        lines.append(
            f'<text x="{left + 22}" y="{cy + 7:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#1f1f1f">{escape(task_label)}</text>'
        )
        lines.append(
            f'<rect x="{chart_x}" y="{bar_y:.1f}" width="{chart_w:.1f}" height="{bar_h:.1f}" rx="8" fill="#efede4"/>'
        )
        lines.append(
            f'<rect x="{chart_x}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="8" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{chart_x + bar_w + 14:.1f}" y="{cy + 7:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#1f1f1f">{rate:.1f}%</text>'
        )
        lines.append(
            f'<text x="{chart_x + bar_w + 92:.1f}" y="{cy + 7:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#555">({escape(counts)})</text>'
        )

    lines.append(
        f'<text x="110" y="{height - 70}" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#151515">Reading</text>'
    )
    lines.append(
        f'<text x="220" y="{height - 70}" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#333">Higher is better. This slide omits episode success and focuses only on partial task progress.</text>'
    )
    lines.append("</svg>")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_slide_heatmap_svg_chart(
    output_path: Path,
    run_id: str,
    heatmap: list[list[float]],
    heatmap_tasks: list[str],
    heatmap_cases: list[str],
) -> None:
    width = 1600
    height = 940
    left = 120
    right = 130
    top = 180
    bottom = 160
    panel_x = left
    panel_y = top
    panel_w = width - left - right
    panel_h = height - top - bottom
    label_w = 180
    legend_w = 20
    legend_gap = 18
    heat_x = panel_x + label_w
    heat_y = panel_y + 30
    heat_w = panel_w - label_w - legend_w - legend_gap - 20
    heat_h = panel_h - 70

    task_labels = [display_label(task) for task in heatmap_tasks]
    case_labels = [short_case_label(case_name) for case_name in heatmap_cases]
    max_rate = max(35.0, max_finite(flatten(heatmap)))
    cell_w = heat_w / len(case_labels)
    cell_h = heat_h / len(task_labels)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf5"/>',
        '<text x="800" y="64" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="700" fill="#151515">OpenVLA per-case subtask completion</text>',
        '<text x="800" y="102" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#474747">LIBERO Spatial | Heatmap by task type and case</text>',
        f'<text x="800" y="128" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#666">Run: {escape(run_id)}</text>',
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="18" fill="#ffffff" stroke="#ddd8ca"/>',
    ]

    for j, case_label in enumerate(case_labels):
        x = heat_x + j * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x:.1f}" y="{heat_y + heat_h + 34:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#555">{escape(case_label)}</text>'
        )

    for i, task_label in enumerate(task_labels):
        y = heat_y + i * cell_h + cell_h / 2 + 8
        lines.append(
            f'<text x="{panel_x + 24}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#1f1f1f">{escape(task_label)}</text>'
        )
        for j, _ in enumerate(case_labels):
            value = heatmap[i][j]
            x = heat_x + j * cell_w
            y0 = heat_y + i * cell_h
            fill = color_for_value(value, max_rate)
            text_color = "#ffffff" if value >= 18 else "#10212b"
            lines.append(
                f'<rect x="{x:.1f}" y="{y0:.1f}" width="{cell_w - 6:.1f}" height="{cell_h - 6:.1f}" rx="8" fill="{fill}" stroke="#ffffff" stroke-width="1.5"/>'
            )
            lines.append(
                f'<text x="{x + (cell_w - 6) / 2:.1f}" y="{y0 + (cell_h - 6) / 2 + 8:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700" fill="{text_color}">{value:.1f}</text>'
            )

    legend_x = heat_x + heat_w + legend_gap
    legend_y = heat_y
    legend_h = heat_h
    for step in range(140):
        ratio0 = step / 140
        y = legend_y + (1 - ratio0) * legend_h
        color = color_for_value(ratio0 * max_rate, max_rate)
        lines.append(
            f'<rect x="{legend_x:.1f}" y="{y:.1f}" width="{legend_w}" height="{legend_h / 140 + 1:.2f}" fill="{color}" stroke="none"/>'
        )
    lines.append(
        f'<text x="{legend_x + legend_w + 10:.1f}" y="{legend_y + 8:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">{max_rate:.0f}</text>'
    )
    lines.append(
        f'<text x="{legend_x + legend_w + 10:.1f}" y="{legend_y + legend_h:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">0</text>'
    )
    lines.append(
        f'<text x="{legend_x + 52:.1f}" y="{legend_y + legend_h / 2:.1f}" transform="rotate(90 {legend_x + 52:.1f},{legend_y + legend_h / 2:.1f})" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">Completion (%)</text>'
    )

    lines.append(
        f'<text x="120" y="{height - 70}" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#151515">Reading</text>'
    )
    lines.append(
        f'<text x="230" y="{height - 70}" font-family="Helvetica, Arial, sans-serif" font-size="20" fill="#333">Darker cells indicate stronger partial progress. Each value is a per-case subtask completion percentage.</text>'
    )
    lines.append("</svg>")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_results(args.json_path)

    summary = data["results_by_task_type"]
    ordered_tasks = [task for task in TASK_ORDER if task in summary]

    subtask_rates = [summary[task]["subtask_rate"] * 100 for task in ordered_tasks]
    numerator_labels = [
        f'{summary[task]["agent_subtasks"]}/{summary[task]["possible_subtasks"]}'
        for task in ordered_tasks
    ]

    heatmap, heatmap_tasks, heatmap_cases = build_heatmap(data["detailed_task_results"])

    info = data["evaluation_info"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.variant == "slide":
        svg_path = args.output if args.output.suffix.lower() == ".svg" else args.output.with_suffix(".svg")
        save_slide_svg_chart(
            svg_path,
            info["run_id"],
            ordered_tasks,
            subtask_rates,
            numerator_labels,
        )
        print(f"Saved chart to {svg_path}")
        return

    if args.variant == "slide_heatmap":
        svg_path = args.output if args.output.suffix.lower() == ".svg" else args.output.with_suffix(".svg")
        save_slide_heatmap_svg_chart(
            svg_path,
            info["run_id"],
            heatmap,
            heatmap_tasks,
            heatmap_cases,
        )
        print(f"Saved chart to {svg_path}")
        return

    if plt is not None and args.output.suffix.lower() != ".svg":
        import numpy as np

        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 15,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.titlesize": 19,
            }
        )

        fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])

        ax_bar = fig.add_subplot(gs[0, 0])
        colors = [TASK_COLORS.get(task, "#777777") for task in ordered_tasks]
        bars = ax_bar.barh(ordered_tasks, subtask_rates, color=colors, edgecolor="#1f1f1f", linewidth=0.6)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max(35, max(subtask_rates) + 5))
        ax_bar.set_xlabel("Subtask completion rate (%)")
        ax_bar.set_title("Aggregate by task type")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.28)
        ax_bar.set_axisbelow(True)

        for bar, rate, counts in zip(bars, subtask_rates, numerator_labels):
            y = bar.get_y() + bar.get_height() / 2
            ax_bar.text(
                bar.get_width() + 0.6,
                y,
                f"{rate:.1f}%  ({counts})",
                va="center",
                ha="left",
                fontsize=10,
                color="#222222",
            )

        ax_heat = fig.add_subplot(gs[0, 1])
        heatmap_np = np.array(heatmap, dtype=float)
        im = ax_heat.imshow(heatmap_np, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(35, np.nanmax(heatmap_np)))
        ax_heat.set_title("Per-case subtask completion (%)")
        ax_heat.set_xticks(np.arange(len(heatmap_cases)))
        ax_heat.set_xticklabels(heatmap_cases)
        ax_heat.set_yticks(np.arange(len(heatmap_tasks)))
        ax_heat.set_yticklabels(heatmap_tasks)
        ax_heat.set_xlabel("Case")

        for i in range(len(heatmap)):
            for j in range(len(heatmap[i])):
                value = heatmap[i][j]
                if math.isnan(value):
                    continue
                text_color = "white" if value >= 18 else "#10212b"
                ax_heat.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9)

        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label("Subtask completion (%)")

        fig.suptitle(
            f"OpenVLA on RoboCerebra LIBERO Spatial\n"
            f"Subtask-centric view for {info['run_id']}",
            x=0.48,
            y=1.02,
        )

        fig.text(
            0.01,
            -0.02,
            "Episode success omitted here because no episode fully completed all goal conditions.\n"
            "Rates show completed subtasks divided by possible subtasks in the evaluator output.",
            fontsize=10,
            color="#333333",
        )

        fig.savefig(args.output, dpi=220, bbox_inches="tight")
        fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved chart to {args.output}")
        print(f"Saved chart to {args.output.with_suffix('.svg')}")
        return

    svg_path = args.output if args.output.suffix.lower() == ".svg" else args.output.with_suffix(".svg")
    save_svg_chart(
        svg_path,
        info["run_id"],
        ordered_tasks,
        subtask_rates,
        numerator_labels,
        heatmap,
        heatmap_tasks,
        heatmap_cases,
    )
    print(f"Saved chart to {svg_path}")


if __name__ == "__main__":
    main()
