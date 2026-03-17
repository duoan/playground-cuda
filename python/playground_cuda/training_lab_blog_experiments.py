"""Run reproducible experiment cases for the training lab blog report."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    slug: str
    title: str
    category: str
    args: tuple[str, ...]


CASES = (
    Case(
        slug="baseline",
        title="Baseline",
        category="baseline",
        args=("--mode", "baseline", "--steps", "20", "--skip-first-steps", "2"),
    ),
    Case(
        slug="data_workers0",
        title="Data bottleneck (workers=0)",
        category="data",
        args=("--mode", "data", "--steps", "12", "--num-workers", "0", "--skip-first-steps", "2"),
    ),
    Case(
        slug="data_workers4",
        title="Data bottleneck fixed (workers=4)",
        category="data",
        args=("--mode", "data", "--steps", "12", "--num-workers", "4", "--skip-first-steps", "2"),
    ),
    Case(
        slug="data_workers8",
        title="Data bottleneck fixed (workers=8)",
        category="data",
        args=("--mode", "data", "--steps", "12", "--num-workers", "8", "--skip-first-steps", "2"),
    ),
    Case(
        slug="torch_eager",
        title="Torch-bound eager",
        category="torch",
        args=("--mode", "torch", "--steps", "20", "--skip-first-steps", "2"),
    ),
    Case(
        slug="torch_batch1024",
        title="Torch-bound larger batch (1024)",
        category="torch",
        args=("--mode", "torch", "--steps", "20", "--batch-size", "1024", "--skip-first-steps", "2"),
    ),
    Case(
        slug="kernel_fp32",
        title="Kernel-bound fp32",
        category="kernel",
        args=("--mode", "kernel", "--steps", "12", "--amp", "none", "--skip-first-steps", "2"),
    ),
    Case(
        slug="kernel_bf16",
        title="Kernel-bound bf16",
        category="kernel",
        args=("--mode", "kernel", "--steps", "12", "--amp", "bf16", "--skip-first-steps", "2"),
    ),
)


def run_case(case: Case, output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / f"{case.slug}.json"
    log_path = output_dir / f"{case.slug}.log"
    command = [
        sys.executable,
        "-m",
        "playground_cuda.training_lab",
        *case.args,
        "--summary-json",
        str(summary_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["slug"] = case.slug
    summary["title"] = case.title
    summary["category"] = case.category
    summary["command"] = " ".join(command)
    return summary


def write_csv(results: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "slug",
        "title",
        "category",
        "mode",
        "compile",
        "amp",
        "batch_size",
        "num_workers",
        "average_step_ms",
        "steady_state_step_ms",
        "step_time_p50_ms",
        "steady_state_samples_per_s",
        "final_loss",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result.get(name) for name in fieldnames})


def make_bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    output_path: Path,
    bar_color: str = "#2a9d8f",
    unit: str = "ms",
) -> None:
    width = 760
    height = 340
    margin_left = 180
    margin_right = 40
    margin_top = 50
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1.0
    bar_gap = 18
    bar_height = max(24, (plot_height - bar_gap * (len(values) - 1)) / max(len(values), 1))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #1f2933; }',
        '.title { font-size: 22px; font-weight: 700; }',
        '.label { font-size: 14px; }',
        '.value { font-size: 14px; font-weight: 700; }',
        '.axis { stroke: #94a3b8; stroke-width: 1; }',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_left}" y="28" class="title">{title}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis" />',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * (bar_height + bar_gap)
        bar_width = 0 if max_value == 0 else (value / max_value) * plot_width
        lines.append(f'<text x="{margin_left - 12}" y="{y + bar_height / 2 + 5:.1f}" text-anchor="end" class="label">{label}</text>')
        lines.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="6" fill="{bar_color}" />'
        )
        lines.append(
            f'<text x="{margin_left + bar_width + 8:.1f}" y="{y + bar_height / 2 + 5:.1f}" class="value">{value:.1f} {unit}</text>'
        )

    for tick_index in range(5):
        tick_value = max_value * tick_index / 4 if max_value else 0
        x = margin_left + plot_width * tick_index / 4
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top - 8}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e2e8f0" />')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="label">{tick_value:.0f}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_line_chart(
    title: str,
    series: list[tuple[str, list[float], str]],
    output_path: Path,
) -> None:
    width = 840
    height = 420
    margin_left = 70
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(values) for _, values, _ in series)
    max_steps = max(len(values) for _, values, _ in series)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #1f2933; }',
        '.title { font-size: 22px; font-weight: 700; }',
        '.label { font-size: 13px; }',
        '.legend { font-size: 13px; font-weight: 700; }',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_left}" y="28" class="title">{title}</text>',
    ]

    for tick in range(5):
        y = margin_top + plot_height * tick / 4
        value = max_value * (1 - tick / 4)
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e2e8f0" />')
        lines.append(f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.0f}</text>')

    for step in range(max_steps):
        x = margin_left + plot_width * step / max(max_steps - 1, 1)
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#f1f5f9" />')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 22}" text-anchor="middle" class="label">{step}</text>')

    legend_x = margin_left
    for name, values, color in series:
        points: list[str] = []
        for step, value in enumerate(values):
            x = margin_left + plot_width * step / max(max_steps - 1, 1)
            y = margin_top + plot_height * (1 - value / max_value)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}" />')
        lines.append(f'<rect x="{legend_x}" y="{height - 32}" width="14" height="14" fill="{color}" rx="3" />')
        lines.append(f'<text x="{legend_x + 20}" y="{height - 20}" class="legend">{name}</text>')
        legend_x += 230

    lines.append(f'<text x="{width / 2:.1f}" y="{height - 6}" text-anchor="middle" class="label">step</text>')
    lines.append(f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 18 {height / 2:.1f})">step time (ms)</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = Path("reports/training_lab_blog")
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    results = [run_case(case, output_dir) for case in CASES]
    write_csv(results, output_dir / "timings.csv")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    data_cases = [result for result in results if result["category"] == "data"]
    torch_cases = [result for result in results if result["category"] == "torch"]
    kernel_cases = [result for result in results if result["category"] == "kernel"]

    make_bar_chart(
        "Data Pipeline Experiments",
        [result["title"] for result in data_cases],
        [float(result["steady_state_step_ms"]) for result in data_cases],
        charts_dir / "data_pipeline.svg",
        bar_color="#e76f51",
    )
    make_line_chart(
        "Data Pipeline Step-Time Jitter",
        [
            (result["title"], [float(value) for value in result["step_times_ms"]], color)
            for result, color in zip(
                data_cases,
                ("#c2410c", "#0284c7", "#16a34a"),
            )
        ],
        charts_dir / "data_jitter.svg",
    )
    make_bar_chart(
        "Torch-Bound Throughput",
        [result["title"] for result in torch_cases],
        [float(result["steady_state_samples_per_s"]) for result in torch_cases],
        charts_dir / "torch_throughput.svg",
        bar_color="#264653",
        unit="samples/s",
    )
    make_bar_chart(
        "Kernel-Bound fp32 vs bf16",
        [result["title"] for result in kernel_cases],
        [float(result["steady_state_step_ms"]) for result in kernel_cases],
        charts_dir / "kernel_amp.svg",
        bar_color="#3a86ff",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
