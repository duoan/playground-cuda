"""Run reproducible transformer token-skew experiments and generate charts."""

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
    strategy: str
    args: tuple[str, ...]


CASES = (
    Case(
        slug="fixed_sample_ws4",
        title="Uniform lengths",
        strategy="sample",
        args=(
            "--strategy",
            "sample",
            "--length-mode",
            "fixed",
            "--fixed-seq-len",
            "128",
            "--steps",
            "12",
        ),
    ),
    Case(
        slug="variable_sample_ws4",
        title="Variable lengths + fixed batch",
        strategy="sample",
        args=(
            "--strategy",
            "sample",
            "--length-mode",
            "variable",
            "--steps",
            "12",
        ),
    ),
    Case(
        slug="variable_bucket_ws4",
        title="Variable lengths + bucketing",
        strategy="bucket",
        args=(
            "--strategy",
            "bucket",
            "--length-mode",
            "variable",
            "--steps",
            "12",
        ),
    ),
    Case(
        slug="variable_token_ws4",
        title="Variable lengths + token budget",
        strategy="token",
        args=(
            "--strategy",
            "token",
            "--length-mode",
            "variable",
            "--max-tokens-per-batch",
            "1280",
            "--steps",
            "12",
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
        "4",
        "-m",
        "playground_cuda.transformer_token_skew_lab",
        *case.args,
        "--summary-json",
        str(summary_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["slug"] = case.slug
    summary["title"] = case.title
    summary["strategy_name"] = case.strategy
    summary["command"] = " ".join(command)
    return summary


def write_csv(results: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "slug",
        "title",
        "strategy",
        "length_mode",
        "slowest_step_ms",
        "global_useful_tokens_per_s",
        "global_padded_tokens_per_s",
        "average_padding_ratio",
        "average_rank_time_skew_ms",
        "average_rank_token_spread",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "slug": result["slug"],
                    "title": result["title"],
                    "strategy": result["strategy"],
                    "length_mode": result["length_mode"],
                    "slowest_step_ms": result["slowest_step_ms"],
                    "global_useful_tokens_per_s": result["global_useful_tokens_per_s"],
                    "global_padded_tokens_per_s": result["global_padded_tokens_per_s"],
                    "average_padding_ratio": result["average_padding_ratio"],
                    "average_rank_time_skew_ms": result["average_rank_time_skew_ms"],
                    "average_rank_token_spread": result["average_rank_token_spread"],
                }
            )


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
    width = 860
    height = 360
    margin_left = 240
    margin_right = 60
    margin_top = 56
    margin_bottom = 64
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
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_left}" y="32" class="title">{title}</text>',
    ]

    for tick in range(5):
        tick_value = max_value * tick / 4 if max_value else 0.0
        x = margin_left + plot_width * tick / 4
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top - 6}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e2e8f0" />')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="label">{tick_value:.0f}</text>')

    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * (bar_height + bar_gap)
        width_value = 0.0 if max_value == 0 else value / max_value * plot_width
        lines.append(f'<text x="{margin_left - 10}" y="{y + bar_height / 2 + 4:.1f}" text-anchor="end" class="label">{label}</text>')
        lines.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{width_value:.1f}" height="{bar_height:.1f}" rx="6" fill="{bar_color}" />')
        lines.append(
            f'<text x="{margin_left + width_value + 8:.1f}" y="{y + bar_height / 2 + 4:.1f}" class="value">{value:.{decimals}f} {unit}</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_rank_work_chart(sample_case: dict[str, object], token_case: dict[str, object], output_path: Path) -> None:
    width = 980
    height = 460
    margin_left = 170
    margin_right = 50
    margin_top = 62
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    sample_useful = sample_case["per_rank_useful_tokens"]
    sample_padded = sample_case["per_rank_padded_tokens"]
    token_useful = token_case["per_rank_useful_tokens"]
    token_padded = token_case["per_rank_padded_tokens"]

    labels = [f"sample r{rank}" for rank in sorted(sample_useful, key=int)] + [
        f"token r{rank}" for rank in sorted(token_useful, key=int)
    ]
    useful_values = [float(sample_useful[rank]) for rank in sorted(sample_useful, key=int)] + [
        float(token_useful[rank]) for rank in sorted(token_useful, key=int)
    ]
    waste_values = [float(sample_padded[rank]) - float(sample_useful[rank]) for rank in sorted(sample_useful, key=int)] + [
        float(token_padded[rank]) - float(token_useful[rank]) for rank in sorted(token_useful, key=int)
    ]
    total_values = [useful + waste for useful, waste in zip(useful_values, waste_values)]
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
        f'<text x="{margin_left}" y="34" class="title">Per-Rank Useful Tokens vs Padding Waste</text>',
    ]

    for tick in range(5):
        tick_value = max_value * tick / 4
        x = margin_left + plot_width * tick / 4
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top - 6}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e2e8f0" />')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="label">{tick_value:.0f}</text>')

    for index, (label, useful_value, waste_value, total_value) in enumerate(zip(labels, useful_values, waste_values, total_values)):
        y = margin_top + index * (bar_height + bar_gap)
        useful_width = 0.0 if max_value == 0 else useful_value / max_value * plot_width
        waste_width = 0.0 if max_value == 0 else waste_value / max_value * plot_width
        lines.append(f'<text x="{margin_left - 10}" y="{y + bar_height / 2 + 4:.1f}" text-anchor="end" class="label">{label}</text>')
        lines.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{useful_width:.1f}" height="{bar_height:.1f}" rx="5" fill="#0f766e" />')
        lines.append(
            f'<rect x="{margin_left + useful_width:.1f}" y="{y:.1f}" width="{waste_width:.1f}" height="{bar_height:.1f}" rx="5" fill="#f97316" />'
        )
        lines.append(
            f'<text x="{margin_left + useful_width + waste_width + 8:.1f}" y="{y + bar_height / 2 + 4:.1f}" class="value">{total_value:.0f}</text>'
        )

    legend_y = height - 26
    lines.append(f'<rect x="{margin_left}" y="{legend_y - 12}" width="16" height="16" rx="3" fill="#0f766e" />')
    lines.append(f'<text x="{margin_left + 22}" y="{legend_y}" class="legend">useful tokens</text>')
    lines.append(f'<rect x="{margin_left + 146}" y="{legend_y - 12}" width="16" height="16" rx="3" fill="#f97316" />')
    lines.append(f'<text x="{margin_left + 168}" y="{legend_y}" class="legend">padding waste</text>')
    lines.append(f'<text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" class="label">tokens per steady-state step</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = Path("reports/transformer_token_skew_blog")
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    results = [run_case(case, output_dir) for case in CASES]
    write_csv(results, output_dir / "timings.csv")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    labels = [result["title"] for result in results]
    make_bar_chart(
        "Variable-Length Transformer: Slowest Step Time",
        labels,
        [float(result["slowest_step_ms"]) for result in results],
        charts_dir / "strategy_step_time.svg",
        bar_color="#dc2626",
        unit="ms",
        decimals=1,
    )
    make_bar_chart(
        "Variable-Length Transformer: Useful Token Throughput",
        labels,
        [float(result["global_useful_tokens_per_s"]) for result in results],
        charts_dir / "strategy_useful_tokens_per_s.svg",
        bar_color="#2563eb",
        unit="tokens/s",
        decimals=0,
    )
    make_bar_chart(
        "Variable-Length Transformer: Average Padding Ratio",
        labels,
        [100.0 * float(result["average_padding_ratio"]) for result in results],
        charts_dir / "strategy_padding_ratio.svg",
        bar_color="#f97316",
        unit="%",
        decimals=1,
    )
    make_bar_chart(
        "Variable-Length Transformer: Average Rank Token Spread",
        labels,
        [float(result["average_rank_token_spread"]) for result in results],
        charts_dir / "strategy_token_spread.svg",
        bar_color="#7c3aed",
        unit="tokens",
        decimals=0,
    )

    by_slug = {result["slug"]: result for result in results}
    make_rank_work_chart(
        by_slug["variable_sample_ws4"],
        by_slug["variable_token_ws4"],
        charts_dir / "sample_vs_token_rank_work.svg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
