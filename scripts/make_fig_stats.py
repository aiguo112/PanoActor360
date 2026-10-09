#!/usr/bin/env python3
"""Plot person-latitude and persons-per-frame histograms from person_meta.

Reads every data/person_meta/*.json. Latitude in those files is radians.
Writes paper_fg2027/figs/stats.pdf at single-column width (3.5 in).
The seam-crossing share is left for the LaTeX caption.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fig_style  # noqa: E402

META_ROOT = REPO_ROOT / "data" / "person_meta"
OUT_PATH = REPO_ROOT / "paper_fg2027" / "figs" / "stats.pdf"
LAT_BIN_DEG = 15
MAX_PERSONS_PER_FRAME = 8
COL_W_IN = 3.5
FIG_H_IN = 2.05


def load_counts(meta_root: Path) -> tuple[np.ndarray, np.ndarray]:
    n_lat = 180 // LAT_BIN_DEG
    lat_counts = np.zeros(n_lat, dtype=np.int64)
    frame_counts = np.zeros(MAX_PERSONS_PER_FRAME + 1, dtype=np.int64)
    paths = sorted(meta_root.glob("*.json"))
    if not paths:
        raise SystemExit(f"no person_meta JSON files in {meta_root}")
    for index, path in enumerate(paths, start=1):
        document = json.loads(path.read_text(encoding="utf-8"))
        for people in document.get("frames") or []:
            frame_counts[min(len(people), MAX_PERSONS_PER_FRAME)] += 1
            for person in people:
                lat_deg = math.degrees(float(person["lat"]))
                bin_index = int((lat_deg + 90.0) // LAT_BIN_DEG)
                if lat_deg >= 90.0:
                    bin_index = n_lat - 1
                bin_index = min(max(bin_index, 0), n_lat - 1)
                lat_counts[bin_index] += 1
        if index % 1000 == 0 or index == len(paths):
            print(f"  read {index}/{len(paths)} meta files", flush=True)
    return lat_counts, frame_counts


def main() -> None:
    fig_style.apply()
    print(f"reading {META_ROOT}", flush=True)
    lat_counts, frame_counts = load_counts(META_ROOT)
    lat_edges = np.arange(-90, 90 + LAT_BIN_DEG, LAT_BIN_DEG)
    lat_centers = lat_edges[:-1] + LAT_BIN_DEG / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(COL_W_IN, FIG_H_IN), layout="constrained")

    axes[0].bar(
        lat_centers,
        lat_counts / 1000.0,
        width=LAT_BIN_DEG * 0.9,
        color=fig_style.SLATE,
        linewidth=0,
        zorder=3,
    )
    axes[0].set_xlim(-90, 90)
    axes[0].set_xticks([-90, -45, 0, 45, 90])
    axes[0].set_xlabel("Latitude (°)")
    axes[0].set_ylabel("Detections (×10³)")
    axes[0].set_title("(a) Person latitude", loc="left")

    colors = [fig_style.MUTED] * MAX_PERSONS_PER_FRAME + [fig_style.ACCENT]
    axes[1].bar(
        np.arange(MAX_PERSONS_PER_FRAME + 1),
        frame_counts / 1000.0,
        width=0.78,
        color=colors,
        linewidth=0,
        zorder=3,
    )
    axes[1].set_xlim(-0.6, MAX_PERSONS_PER_FRAME + 0.6)
    axes[1].set_xticks(range(MAX_PERSONS_PER_FRAME + 1))
    axes[1].set_xlabel("Persons per frame")
    axes[1].set_ylabel("Frames (×10³)")
    axes[1].set_title("(b) Crowding", loc="left")

    for ax in axes:
        fig_style.polish(ax, grid="y")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PATH.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(
        f"wrote {OUT_PATH}  people {int(lat_counts.sum())}  frames {int(frame_counts.sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
