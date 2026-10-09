#!/usr/bin/env python3
"""Pool global DINOv2 tokens from 16x32 down to 8x16.

Single process, no DataLoader. For each feats/global/{seg_id}.pt, average-pool
"tokens" over latitude and longitude and write feats/global8/{seg_id}.pt.
An output that already exists is skipped. Source files are left unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "feats" / "global"
POOLED_ROOT = REPO_ROOT / "feats" / "global8"

N_FRAMES = 16
SRC_H = 16
SRC_W = 32
POOLED_H = 8
POOLED_W = 16
FEAT_DIM = 768


def save_pt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    values = tokens.detach().to(device="cpu", dtype=torch.float32)
    if tuple(values.shape) != (N_FRAMES, SRC_H, SRC_W, FEAT_DIM):
        raise RuntimeError(f"expected tokens {(N_FRAMES, SRC_H, SRC_W, FEAT_DIM)}, got {tuple(values.shape)}")
    pooled = values.reshape(N_FRAMES, POOLED_H, 2, POOLED_W, 2, FEAT_DIM).mean(dim=(2, 4))
    return pooled.to(dtype=torch.float16)


def segment_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*.pt"), key=lambda path: int(path.stem))


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"missing source directory: {SOURCE_ROOT}")
    paths = segment_paths(SOURCE_ROOT)
    POOLED_ROOT.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    total = len(paths)
    print(f"pooling {total} segments into {POOLED_ROOT}", flush=True)
    for index, path in enumerate(paths, start=1):
        destination = POOLED_ROOT / f"{path.stem}.pt"
        if destination.is_file():
            skipped += 1
        else:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            cls = blob["cls"].detach().cpu()
            if tuple(cls.shape) != (N_FRAMES, FEAT_DIM):
                raise RuntimeError(f"{path.name} cls has shape {tuple(cls.shape)}")
            save_pt(destination, {"tokens": pool_tokens(blob["tokens"]), "cls": cls})
            written += 1
        if index % 200 == 0 or index == total:
            print(f"  {index}/{total}  written {written}  skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
