#!/usr/bin/env python3
"""Per-fold top-1 grouped bars — clearer than overlapping line charts."""

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

CSV_PATH = REPO_ROOT / "results" / "perfold_main.csv"
OUT_PDF = REPO_ROOT / "paper_fg2027" / "figs" / "perfold.pdf"

# Few methods so each fold stays readable in one column / full width.
CONFIGS = (
    "erp_cls_mean",
    "cubemap",
    "grid_latlon",
    "persons_only",
    "full",
)


def load() -> dict[str, list[float]]:
    if not CSV_PATH.is_file():
        raise SystemExit(f"missing {CSV_PATH}")
    data: dict[str, dict[int, float]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data.setdefault(row["model"], {})[int(row["fold"])] = float(row["top1"])
    out: dict[str, list[float]] = {}
    for config in CONFIGS:
        if config not in data or len(data[config]) != 5:
            continue
        out[config] = [data[config][fold] for fold in range(5)]
    if not out:
        raise SystemExit("no per-fold series")
    return out


def main() -> None:
    fig_style.apply()
    series = load()
    configs = [c for c in CONFIGS if c in series]
    folds = np.arange(5)
    width = 0.15
    offsets = (np.arange(len(configs)) - (len(configs) - 1) / 2.0) * width

    figure, ax = plt.subplots(figsize=(7.0, 3.35), layout="constrained")
    for offset, config in zip(offsets, configs):
        vals = series[config]
        bars = ax.bar(
            folds + offset,
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

    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(20, 45)
    ax.set_xlabel("")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor="#D1D5DB",
        framealpha=1.0,
        handlelength=1.6,
        columnspacing=1.4,
        handletextpad=0.5,
        borderpad=0.55,
    )
    fig_style.polish(ax, grid="y")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    figure.savefig(OUT_PDF.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(figure)
    print(f"wrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
