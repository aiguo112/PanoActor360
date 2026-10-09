#!/usr/bin/env python3
"""Subset figure: absolute top-1 by condition (grouped bars) + overall ranking.

Uses Times New Roman via fig_style. Far clips (n=87) omitted (high variance).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fig_style  # noqa: E402

CSV_PATH = REPO_ROOT / "results" / "subset_eval.csv"
OUT_PDF = REPO_ROOT / "paper_fg2027" / "figs" / "subsets.pdf"

CONDITIONS = ("SEAM", "NO_SEAM", "NEAR")
COND_LABELS = {
    "SEAM": "Seam",
    "NO_SEAM": "No-seam",
    "NEAR": "Near-field",
}
MODELS = (
    "erp_cls_mean",
    "grid_plain",
    "persons_only",
    "full",
)


def load() -> dict[str, dict[str, float]]:
    if not CSV_PATH.is_file():
        raise SystemExit(f"missing {CSV_PATH}")
    out: dict[str, dict[str, float]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in MODELS:
                continue
            if row["subset"] not in CONDITIONS and row["subset"] != "ALL":
                continue
            out.setdefault(row["model"], {})[row["subset"]] = float(row["top1_mean"])
    return out


def condition_bars(ax, data: dict[str, dict[str, float]], n_by_cond: dict[str, int]) -> None:
    x = np.arange(len(CONDITIONS))
    width = 0.18
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0) * width
    for offset, config in zip(offsets, MODELS):
        vals = [data[config][cond] for cond in CONDITIONS]
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=fig_style.color(config),
            edgecolor="white",
            linewidth=0.4,
            label=fig_style.label(config),
            zorder=3,
        )
        if config == "full":
            for bar in bars:
                bar.set_edgecolor(fig_style.INK)
                bar.set_linewidth(0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{COND_LABELS[c]}\n($n$={n_by_cond.get(c, 0):,})" for c in CONDITIONS]
    )
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(24, 46)
    ax.set_title("Accuracy by scene condition", loc="left")
    fig_style.polish(ax, grid="y")
    ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor="#D1D5DB",
        framealpha=1.0,
        handlelength=1.6,
        columnspacing=1.3,
        handletextpad=0.5,
        borderpad=0.5,
    )


def overall_bars(ax, data: dict[str, dict[str, float]]) -> None:
    configs = list(MODELS)
    vals = [data[c]["ALL"] for c in configs]
    y = np.arange(len(configs))
    ax.barh(y, vals, color=[fig_style.color(c) for c in configs], height=0.62, linewidth=0, zorder=3)
    for yi, val, config in zip(y, vals, configs):
        ax.text(
            val + 0.3,
            yi,
            f"{val:.1f}",
            va="center",
            ha="left",
            fontsize=8,
            color=fig_style.INK,
            fontweight="bold" if config == "full" else "normal",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([fig_style.label(c) for c in configs])
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_xlim(24, 46)
    ax.set_title("Overall (all clips)", loc="left")
    ax.invert_yaxis()
    fig_style.polish(ax, grid="x")


def radar(ax, data: dict[str, dict[str, float]]) -> None:
    """Tight-range radar so method gaps remain visible."""
    axes = ("SEAM", "NO_SEAM", "NEAR", "ALL")
    labels = ["Seam", "No-seam", "Near", "All"]
    n = len(axes)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(28, 44)
    ax.set_yticks([30, 34, 38, 42])
    ax.set_yticklabels(["30", "34", "38", "42"], fontsize=7, color=fig_style.MUTED)
    ax.xaxis.grid(True, color=fig_style.GRID, linewidth=0.8)
    ax.yaxis.grid(True, color=fig_style.GRID, linewidth=0.8)
    ax.spines["polar"].set_color(fig_style.INK)

    for config in MODELS:
        values = [data[config][key] for key in axes]
        values = values + values[:1]
        ax.plot(
            angles,
            values,
            color=fig_style.color(config),
            linewidth=2.3 if config == "full" else 1.4,
            label=fig_style.label(config),
            zorder=4 if config == "full" else 3,
        )
        ax.fill(angles, values, color=fig_style.color(config), alpha=0.14 if config == "full" else 0.04)


def main() -> None:
    fig_style.apply()
    data = load()
    for config in MODELS:
        need = list(CONDITIONS) + ["ALL"]
        missing = [key for key in need if key not in data.get(config, {})]
        if missing:
            raise SystemExit(f"{config} missing: {missing}")

    # Sample sizes from CSV (full model rows).
    n_by_cond: dict[str, int] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "full" and row["subset"] in CONDITIONS:
                n_by_cond[row["subset"]] = int(float(row["n"]))

    figure = plt.figure(figsize=(7.3, 3.4), layout="constrained")
    ax_bar = figure.add_subplot(1, 2, 1)
    ax_radar = figure.add_subplot(1, 2, 2, projection="polar")
    condition_bars(ax_bar, data, n_by_cond)
    radar(ax_radar, data)
    ax_radar.set_title("Same scores (radar view)", loc="left", pad=16)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    figure.savefig(OUT_PDF.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(figure)
    print(f"wrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
