"""Shared IEEE FG 2027 figure style: Times New Roman, print-safe colors."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Method colors: ours emphasized; baselines cool slate/teal.
PALETTE = {
    "full": "#B45309",          # warm amber-brown — PanoActor
    "full_light": "#D97706",    # lighter — Lite
    "persons_only": "#0F766E",  # teal
    "grid_latlon": "#1E3A5F",   # deep navy
    "grid_plain": "#3B82A0",    # steel
    "erp_cls_mean": "#6B7280",  # gray
    "erp_cls_temporal": "#9CA3AF",
    "cubemap": "#7C2D12",       # dark rust (distinct, not purple-glow)
    "videomae": "#334155",
    "clip_person": "#64748B",
    "siglip_person": "#94A3B8",
    "abl_naive": "#9F1239",
    "abl_no_lat": "#64748B",
    "abl_lin_lon": "#64748B",
    "abl_no_area": "#64748B",
    "abl_no_mil": "#64748B",
    "abl_no_roll": "#64748B",
}

LABELS = {
    "erp_cls_mean": "ERP-CLS",
    "erp_cls_temporal": "ERP-CLS+temp.",
    "cubemap": "Cubemap",
    "grid_plain": "ERP grid",
    "grid_latlon": "ERP grid+sph.",
    "persons_only": "Person stream",
    "full": "PanoActor",
    "full_light": "PanoActor-Lite",
    "videomae": "VideoMAE",
    "clip_person": "CLIP (person)",
    "siglip_person": "SigLIP (person)",
    "abl_no_lat": "w/o latitude",
    "abl_lin_lon": "linear longitude",
    "abl_no_area": "w/o area pool",
    "abl_naive": "naive ERP crops",
    "abl_no_mil": "w/o actor attn.",
    "abl_no_roll": "w/o yaw roll",
}

INK = "#111827"
MUTED = "#4B5563"
GRID = "#E5E7EB"
ACCENT = "#B45309"
TEAL = "#0F766E"
SLATE = "#1E3A5F"


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.4,
            "legend.labelspacing": 0.35,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def color(config: str) -> str:
    return PALETTE.get(config, MUTED)


def label(config: str) -> str:
    return LABELS.get(config, config)


def polish(ax, grid: str = "y") -> None:
    ax.tick_params(length=3, width=0.7, pad=2)
    if grid == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    elif grid == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    elif grid == "both":
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
