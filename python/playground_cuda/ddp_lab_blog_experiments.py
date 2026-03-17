"""Run reproducible DDP lab experiments and generate blog-ready charts."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    slug: str
    title: str
    category: str
    world_size: int
    args: tuple[str, ...]


CASES = (
    Case(
        slug="weak_ws1",
        title="Weak scaling ws=1",
        category="weak",
        world_size=1,
        args=("--mode", "baseline", "--steps", "12", "--batch-size", "128", "--skip-first-steps", "2"),
    ),
    Case(
        slug="weak_ws2",
        title="Weak scaling ws=2",
        category="weak",
        world_size=2,
        args=("--mode", "baseline", "--steps", "12", "--batch-size", "128", "--skip-first-steps", "2"),
    ),
    Case(
        slug="weak_ws4",
        title="Weak scaling ws=4",
        category="weak",
        world_size=4,
        args=("--mode", "baseline", "--steps", "12", "--batch-size", "128", "--skip-first-steps", "2"),
    ),
    Case(
        slug="strong_ws1",
        title="Strong scaling ws=1",
        category="strong",
        world_size=1,
        args=("--mode", "baseline", "--steps", "12", "--batch-size", "512", "--skip-first-steps", "2"),
    ),
    Case(
        slug="strong_ws2",
        title="Strong scaling ws=2",
        category="strong",
        world_size=2,
        args=("--mode", "baseline", "--steps", "12", "--batch-size", "256", "--skip-first-steps", "2"),
    ),
    Case(
        slug="skew_ws4",
        title="Straggler rank",
        category="pathology",
        world_size=4,
        args=(
            "--mode",
            "skew",
            "--steps",
            "12",
            "--batch-size",
            "128",
            "--sleep-ms",
            "40",
            "--skip-first-steps",
            "2",
        ),
    ),
    Case(
        slug="comm_ws4",
        title="Extra all-reduce",
        category="pathology",
        world_size=4,
        args=(
            "--mode",
            "comm",
            "--steps",
            "12",
            "--batch-size",
            "128",
            "--comm-mb",
            "128",
            "--skip-first-steps",
            "2",
        ),
    ),
    Case(
        slug="barrier_ws4",
        title="Rank-0 work + barrier",
        category="pathology",
        world_size=4,
        args=(
            "--mode",
            "barrier",
            "--steps",
            "12",
            "--batch-size",
            "128",
            "--barrier-every",
            "1",
            "--barrier-sleep-ms",
            "20",
            "--skip-first-steps",
            "2",
        ),
    ),
)


def run_case(case: Case, output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / f"{case.slug}.json"
    log_path = output_dir / f"{case.slug}.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(case.world_size),
        "-m",
        "playground_cuda.ddp_lab",
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
        "world_size",
        "batch_size",
        "global_batch_size",
        "slowest_rank",
        "slowest_step_ms",
        "global_throughput_samples_per_s",
        "average_rank_skew_ms",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result.get(name) for name in fieldnames})


def rank_keys(rank_map: dict[str, float]) -> list[str]:
    return sorted(rank_map, key=int)


def make_bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    output_path: Path,
    *,
    bar_color: str,
    unit: str,
    decimals: int = 1,
) -> None:
    width = 820
    height = 360
    margin_left = 220
    margin_right = 60
    margin_top = 54
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1.0
    bar_gap = 16
    bar_height = max(24, (plot_height - bar_gap * max(len(values) - 1, 0)) / max(len(values), 1))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #17212b; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".label { font-size: 14px; }",
        ".value { font-size: 14px; font-weight: 700; }",
        ".axis { stroke: #94a3b8; stroke-width: 1; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_left}" y="30" class="title">{title}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis" />',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * (bar_height + bar_gap)
        bar_width = 0.0 if max_value == 0 else (value / max_value) * plot_width
        lines.append(f'<text x="{margin_left - 12}" y="{y + bar_height / 2 + 5:.1f}" text-anchor="end" class="label">{label}</text>')
        lines.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="6" fill="{bar_color}" />'
        )
        lines.append(
            f'<text x="{margin_left + bar_width + 8:.1f}" y="{y + bar_height / 2 + 5:.1f}" class="value">{value:.{decimals}f} {unit}</text>'
        )

    for tick_index in range(5):
        tick_value = max_value * tick_index / 4 if max_value else 0.0
        x = margin_left + plot_width * tick_index / 4
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top - 8}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e2e8f0" />')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="label">{tick_value:.0f}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_scaling_chart(
    title: str,
    x_values: list[int],
    observed: list[float],
    ideal: list[float],
    output_path: Path,
    *,
    y_label: str,
    observed_color: str,
    ideal_color: str,
) -> None:
    width = 860
    height = 420
    margin_left = 80
    margin_right = 32
    margin_top = 58
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(observed), max(ideal))

    def x_pos(world_size: int) -> float:
        return margin_left + plot_width * (world_size - min(x_values)) / max(max(x_values) - min(x_values), 1)

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1 - value / max_value)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #17212b; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".label { font-size: 13px; }",
        ".legend { font-size: 13px; font-weight: 700; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_left}" y="30" class="title">{title}</text>',
    ]

    for tick in range(5):
        y = margin_top + plot_height * tick / 4
        value = max_value * (1 - tick / 4)
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e2e8f0" />')
        lines.append(f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.0f}</text>')

    for world_size in x_values:
        x = x_pos(world_size)
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#f1f5f9" />')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 22}" text-anchor="middle" class="label">{world_size}</text>')

    def polyline(values: list[float], color: str, dashed: bool) -> str:
        points = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in zip(x_values, values))
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash} points="{points}" />'

    lines.append(polyline(ideal, ideal_color, True))
    lines.append(polyline(observed, observed_color, False))

    for x_value, value in zip(x_values, observed):
        lines.append(f'<circle cx="{x_pos(x_value):.1f}" cy="{y_pos(value):.1f}" r="5" fill="{observed_color}" />')

    legend_y = height - 26
    lines.append(f'<rect x="{margin_left}" y="{legend_y - 12}" width="16" height="16" rx="3" fill="{observed_color}" />')
    lines.append(f'<text x="{margin_left + 22}" y="{legend_y}" class="legend">observed</text>')
    lines.append(
        f'<line x1="{margin_left + 126}" y1="{legend_y - 4}" x2="{margin_left + 154}" y2="{legend_y - 4}" '
        f'stroke="{ideal_color}" stroke-width="3" stroke-dasharray="8 6" />'
    )
    lines.append(f'<text x="{margin_left + 162}" y="{legend_y}" class="legend">ideal</text>')

    lines.append(f'<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" class="label">world size</text>')
    lines.append(
        f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 22 {height / 2:.1f})">{y_label}</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_grouped_rank_chart(
    title: str,
    baseline: dict[str, object],
    skew: dict[str, object],
    output_path: Path,
) -> None:
    width = 920
    height = 460
    margin_left = 160
    margin_right = 50
    margin_top = 60
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    baseline_data = baseline["per_rank_data_ms"]
    baseline_compute = baseline["per_rank_compute_ms"]
    skew_data = skew["per_rank_data_ms"]
    skew_compute = skew["per_rank_compute_ms"]

    labels = [f"baseline r{rank}" for rank in rank_keys(baseline_data)] + [f"skew r{rank}" for rank in rank_keys(skew_data)]
    data_values = [float(baseline_data[rank]) for rank in rank_keys(baseline_data)] + [
        float(skew_data[rank]) for rank in rank_keys(skew_data)
    ]
    compute_values = [float(baseline_compute[rank]) for rank in rank_keys(baseline_compute)] + [
        float(skew_compute[rank]) for rank in rank_keys(skew_compute)
    ]
    total_values = [data + compute for data, compute in zip(data_values, compute_values)]
    max_value = max(total_values) if total_values else 1.0
    bar_gap = 14
    bar_height = max(22, (plot_height - bar_gap * max(len(labels) - 1, 0)) / max(len(labels), 1))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #17212b; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".label { font-size: 13px; }",
        ".legend { font-size: 13px; font-weight: 700; }",
        ".value { font-size: 13px; font-weight: 700; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_left}" y="32" class="title">{title}</text>',
    ]

    for tick in range(5):
        tick_value = max_value * tick / 4
        x = margin_left + plot_width * tick / 4
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top - 6}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e2e8f0" />')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="label">{tick_value:.0f}</text>')

    for index, (label, data_value, compute_value, total_value) in enumerate(zip(labels, data_values, compute_values, total_values)):
        y = margin_top + index * (bar_height + bar_gap)
        data_width = 0.0 if max_value == 0 else data_value / max_value * plot_width
        compute_width = 0.0 if max_value == 0 else compute_value / max_value * plot_width
        lines.append(f'<text x="{margin_left - 10}" y="{y + bar_height / 2 + 4:.1f}" text-anchor="end" class="label">{label}</text>')
        lines.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{data_width:.1f}" height="{bar_height:.1f}" rx="5" fill="#f97316" />')
        lines.append(
            f'<rect x="{margin_left + data_width:.1f}" y="{y:.1f}" width="{compute_width:.1f}" '
            f'height="{bar_height:.1f}" rx="5" fill="#2563eb" />'
        )
        lines.append(
            f'<text x="{margin_left + data_width + compute_width + 8:.1f}" y="{y + bar_height / 2 + 4:.1f}" class="value">{total_value:.1f} ms</text>'
        )

    legend_y = height - 26
    lines.append(f'<rect x="{margin_left}" y="{legend_y - 12}" width="16" height="16" rx="3" fill="#f97316" />')
    lines.append(f'<text x="{margin_left + 22}" y="{legend_y}" class="legend">data wait</text>')
    lines.append(f'<rect x="{margin_left + 120}" y="{legend_y - 12}" width="16" height="16" rx="3" fill="#2563eb" />')
    lines.append(f'<text x="{margin_left + 142}" y="{legend_y}" class="legend">compute + sync</text>')
    lines.append(f'<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" class="label">steady-state time (ms)</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def enrich_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    by_slug = {result["slug"]: result for result in results}

    weak_base = float(by_slug["weak_ws1"]["global_throughput_samples_per_s"])
    for slug in ("weak_ws1", "weak_ws2", "weak_ws4"):
        result = by_slug[slug]
        world_size = int(result["world_size"])
        throughput = float(result["global_throughput_samples_per_s"])
        result["weak_scaling_efficiency"] = throughput / (weak_base * world_size)

    strong_base = float(by_slug["strong_ws1"]["global_throughput_samples_per_s"])
    for slug in ("strong_ws1", "strong_ws2", "weak_ws4"):
        result = by_slug[slug]
        world_size = int(result["world_size"])
        throughput = float(result["global_throughput_samples_per_s"])
        result["strong_scaling_efficiency"] = throughput / (strong_base * world_size)

    return results


def main() -> int:
    output_dir = Path("reports/ddp_lab_blog")
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    results = enrich_results([run_case(case, output_dir) for case in CASES])
    write_csv(results, output_dir / "timings.csv")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    by_slug = {result["slug"]: result for result in results}

    weak_cases = [by_slug["weak_ws1"], by_slug["weak_ws2"], by_slug["weak_ws4"]]
    weak_x = [int(result["world_size"]) for result in weak_cases]
    weak_y = [float(result["global_throughput_samples_per_s"]) for result in weak_cases]
    weak_ideal = [weak_y[0] * world_size for world_size in weak_x]
    make_scaling_chart(
        "Weak Scaling Throughput",
        weak_x,
        weak_y,
        weak_ideal,
        charts_dir / "weak_scaling_throughput.svg",
        y_label="global throughput (samples/s)",
        observed_color="#0f766e",
        ideal_color="#94a3b8",
    )

    strong_cases = [by_slug["strong_ws1"], by_slug["strong_ws2"], by_slug["weak_ws4"]]
    strong_labels = [f"ws={int(result['world_size'])}" for result in strong_cases]
    strong_efficiency = [100.0 * float(result["strong_scaling_efficiency"]) for result in strong_cases]
    make_bar_chart(
        "Strong Scaling Efficiency",
        strong_labels,
        strong_efficiency,
        charts_dir / "strong_scaling_efficiency.svg",
        bar_color="#7c3aed",
        unit="%",
        decimals=1,
    )

    pathology_cases = [by_slug["weak_ws4"], by_slug["skew_ws4"], by_slug["comm_ws4"], by_slug["barrier_ws4"]]
    pathology_labels = ["Baseline", "Straggler", "Extra all-reduce", "Rank-0 work + barrier"]
    make_bar_chart(
        "Pathology Sweep: Slowest Step Time",
        pathology_labels,
        [float(result["slowest_step_ms"]) for result in pathology_cases],
        charts_dir / "pathology_slowest_step.svg",
        bar_color="#dc2626",
        unit="ms",
        decimals=1,
    )
    make_bar_chart(
        "Pathology Sweep: Global Throughput",
        pathology_labels,
        [float(result["global_throughput_samples_per_s"]) for result in pathology_cases],
        charts_dir / "pathology_throughput.svg",
        bar_color="#ea580c",
        unit="samples/s",
        decimals=0,
    )
    make_grouped_rank_chart(
        "Why One Slow Rank Hurts Everyone",
        by_slug["weak_ws4"],
        by_slug["skew_ws4"],
        charts_dir / "straggler_rank_breakdown.svg",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
