#!/usr/bin/env python3
"""Main-results ranking and ablation deltas (supplement), Times New Roman."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fig_style  # noqa: E402

RESULTS = REPO_ROOT / "results"
OUT_MAIN = REPO_ROOT / "paper_fg2027" / "figs" / "main_bars.pdf"
OUT_ABL = REPO_ROOT / "paper_fg2027" / "figs" / "ablation_delta.pdf"

MAIN = (
    "erp_cls_mean",
    "erp_cls_temporal",
    "cubemap",
    "grid_plain",
    "grid_latlon",
    "persons_only",
    "full",
    "full_light",
    "videomae",
    "clip_person",
    "siglip_person",
)
ZEROSHOT = {
    "clip_person": ("clip", "person"),
    "siglip_person": ("siglip", "person"),
}
ABL = (
    "full",
    "abl_no_lat",
    "abl_lin_lon",
    "abl_no_area",
    "abl_naive",
    "abl_no_mil",
    "abl_no_roll",
)


def summary(config: str) -> dict | None:
    path = RESULTS / config / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def zeroshot_lookup() -> dict[tuple[str, str], dict]:
    path = RESULTS / "zeroshot.csv"
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[(row["model"], row["view"])] = row
    return out


def main_bars() -> None:
    fig_style.apply()
    rows = []
    zs = zeroshot_lookup()
    for config in MAIN:
        if config in ZEROSHOT:
            payload = zs.get(ZEROSHOT[config])
            if payload is None:
                continue
            rows.append(
                {
                    "config": config,
                    "top1": float(payload["top1_mean"]),
                    "top1_std": float(payload["top1_std"]),
                    "mca": float(payload["mca_mean"]),
                    "mca_std": float(payload["mca_std"]),
                }
            )
            continue
        payload = summary(config)
        if payload is None:
            continue
        rows.append(
            {
                "config": config,
                "top1": float(payload["top1_mean"]),
                "top1_std": float(payload["top1_std"]),
                "mca": float(payload["mca_mean"]),
                "mca_std": float(payload["mca_std"]),
            }
        )
    if not rows:
        return

    # Rank by top-1 so the story reads top-to-bottom.
    rows = sorted(rows, key=lambda row: row["top1"], reverse=True)
    y = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True, layout="constrained")
    colors = [fig_style.color(row["config"]) for row in rows]
    names = [fig_style.label(row["config"]) for row in rows]

    axes[0].barh(
        y,
        [row["top1"] for row in rows],
        xerr=[row["top1_std"] for row in rows],
        color=colors,
        height=0.72,
        linewidth=0,
        error_kw={"ecolor": fig_style.INK, "elinewidth": 0.9, "capsize": 2.0},
        zorder=3,
    )
    axes[1].barh(
        y,
        [row["mca"] for row in rows],
        xerr=[row["mca_std"] for row in rows],
        color=colors,
        height=0.72,
        linewidth=0,
        error_kw={"ecolor": fig_style.INK, "elinewidth": 0.9, "capsize": 2.0},
        zorder=3,
    )
    full = next((row for row in rows if row["config"] == "full"), None)
    for ax, title, xlab, key in (
        (axes[0], "Top-1 accuracy", "Top-1 (%)", "top1"),
        (axes[1], "Mean-class accuracy", "MCA (%)", "mca"),
    ):
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlab)
        if full is not None:
            ax.axvline(full[key], color=fig_style.ACCENT, linestyle="--", linewidth=1.0, alpha=0.75)
        fig_style.polish(ax, grid="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()

    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_MAIN, bbox_inches="tight", dpi=300)
    figure.savefig(OUT_MAIN.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(figure)
    print(f"wrote {OUT_MAIN}", flush=True)


def ablation_delta() -> None:
    fig_style.apply()
    base = summary("full")
    if base is None:
        return
    base_top = float(base["top1_mean"])
    base_mca = float(base["mca_mean"])
    rows = []
    for config in ABL:
        payload = summary(config)
        if payload is None:
            continue
        rows.append(
            {
                "config": config,
                "d_top": float(payload["top1_mean"]) - base_top,
                "d_mca": float(payload["mca_mean"]) - base_mca,
            }
        )
    if not rows:
        return

    y = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, layout="constrained")
    names = ["PanoActor" if row["config"] == "full" else fig_style.label(row["config"]) for row in rows]

    for ax, key, title in (
        (axes[0], "d_top", "Δ top-1 vs PanoActor"),
        (axes[1], "d_mca", "Δ MCA vs PanoActor"),
    ):
        vals = np.array([row[key] for row in rows])
        colors = [
            fig_style.ACCENT if abs(v) < 1e-9 else (fig_style.TEAL if v >= 0 else "#9F1239")
            for v in vals
        ]
        ax.barh(y, vals, color=colors, height=0.68, linewidth=0, zorder=3)
        ax.axvline(0.0, color=fig_style.INK, linewidth=1.0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Percentage points")
        fig_style.polish(ax, grid="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()

    figure.savefig(OUT_ABL, bbox_inches="tight", dpi=300)
    figure.savefig(OUT_ABL.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(figure)
    print(f"wrote {OUT_ABL}", flush=True)


def main() -> None:
    main_bars()
    ablation_delta()


if __name__ == "__main__":
    main()
