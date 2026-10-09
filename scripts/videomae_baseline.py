#!/usr/bin/env python3
"""Frozen VideoMAE (Kinetics) + linear head on ERP clips (T9).

Caches 768-d clip features under feats/videomae/{seg_id}.pt, then trains a
linear classifier with the same 5 folds / seed / recipe style as Table I
(AdamW, 60 epochs, class-balanced CE). Writes results/videomae/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, VideoMAEModel

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "outputs" / "protocol" / "segments_v1.csv"
FRAMES = REPO_ROOT / "data" / "frames_2048"
FEAT_DIR = REPO_ROOT / "feats" / "videomae"
OUT_DIR = REPO_ROOT / "results" / "videomae"
# Pretrained VideoMAE encoder (Kinetics MAE). Loads cleanly into VideoMAEModel.
MODEL_ID = "MCG-NJU/videomae-base"
N_FRAMES = 16
N_CLASSES = 27
SEED = 42
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 0.05
BATCH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--batch-extract", type=int, default=2, help="Clips per forward (VRAM)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_protocol() -> list[dict]:
    rows = []
    with PROTOCOL.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seg_id = str(int(row["seg_id"]))
            if not (FRAMES / seg_id).is_dir():
                continue
            rows.append(
                {
                    "seg_id": seg_id,
                    "video_id": row["video_id"],
                    "class_idx": int(row["class_idx"]),
                    "fold": int(row["fold"]),
                }
            )
    rows.sort(key=lambda row: int(row["seg_id"]))
    return rows


def load_clip_frames(seg_id: str) -> list[np.ndarray]:
    paths = [FRAMES / seg_id / f"{index:02d}.jpg" for index in range(N_FRAMES)]
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Perspective video models expect square frames; resize ERP to 224x224.
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    return frames


@torch.inference_mode()
def extract_features(rows: list[dict], device: torch.device, batch_clips: int) -> None:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not (FEAT_DIR / f"{row['seg_id']}.pt").is_file()]
    print(f"VideoMAE extract: {len(pending)} pending / {len(rows)} total", flush=True)
    if not pending:
        return
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEModel.from_pretrained(MODEL_ID).to(device).eval()
    for start in range(0, len(pending), batch_clips):
        chunk = pending[start : start + batch_clips]
        videos = [load_clip_frames(row["seg_id"]) for row in chunk]
        inputs = processor(videos, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)
        feats = outputs.last_hidden_state.mean(dim=1).cpu()
        for index, row in enumerate(chunk):
            path = FEAT_DIR / f"{row['seg_id']}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save({"feat": feats[index].half(), "seg_id": row["seg_id"]}, temporary)
            os.replace(temporary, path)
        done = min(start + batch_clips, len(pending))
        if done % 20 == 0 or done == len(pending):
            print(f"  extracted {done}/{len(pending)}", flush=True)
    del model
    torch.cuda.empty_cache()


class FeatDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.feats = []
        for row in rows:
            payload = torch.load(FEAT_DIR / f"{row['seg_id']}.pt", map_location="cpu", weights_only=True)
            self.feats.append(payload["feat"].float())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {
            "feat": self.feats[index],
            "label": self.rows[index]["class_idx"],
            "seg_id": self.rows[index]["seg_id"],
        }


def class_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for row in rows:
        counts[row["class_idx"]] += 1
    present = counts > 0
    weights = np.zeros(N_CLASSES, dtype=np.float64)
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def epoch_lr(epoch: int) -> float:
    warmup = 5
    if epoch < warmup:
        return LR * float(epoch + 1) / float(warmup)
    span = EPOCHS - warmup
    progress = (epoch - warmup) / float(max(span, 1))
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))


import math


@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[dict, list[dict]]:
    model.eval()
    logits_all = []
    labels_all = []
    seg_ids = []
    for batch in loader:
        feats = batch["feat"].to(device)
        labels = batch["label"].to(device)
        logits = model(feats)
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
        seg_ids.extend(batch["seg_id"])
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    pred = logits.argmax(dim=-1)
    top1 = 100.0 * float((pred == labels).float().mean())
    top5 = logits.topk(min(5, N_CLASSES), dim=-1).indices
    top5_acc = 100.0 * float((top5 == labels[:, None]).any(dim=-1).float().mean())
    per = []
    for class_idx in labels.unique().tolist():
        mask = labels == class_idx
        per.append(float((pred[mask] == labels[mask]).float().mean()))
    mca = 100.0 * float(np.mean(per)) if per else 0.0
    preds = [
        {
            "seg_id": str(seg_id),
            "true": int(labels[i]),
            "pred": int(pred[i]),
            "top5": [int(x) for x in top5[i].tolist()],
        }
        for i, seg_id in enumerate(seg_ids)
    ]
    return {"top1": top1, "top5": top5_acc, "mca": mca, "n": int(labels.shape[0])}, preds


def train_folds(rows: list[dict], device: torch.device) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = [row for row in rows if (FEAT_DIR / f"{row['seg_id']}.pt").is_file()]
    print(f"train on {len(kept)} clips with cached VideoMAE features", flush=True)
    fold_rows = []
    for fold in range(5):
        set_seed(SEED)
        train_rows = [row for row in kept if row["fold"] != fold]
        test_rows = [row for row in kept if row["fold"] == fold]
        train_loader = DataLoader(FeatDataset(train_rows), batch_size=BATCH, shuffle=True, num_workers=0)
        test_loader = DataLoader(FeatDataset(test_rows), batch_size=BATCH, shuffle=False, num_workers=0)
        model = nn.Linear(768, N_CLASSES).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device), label_smoothing=0.1)
        print(f"fold {fold} train {len(train_rows)} test {len(test_rows)}", flush=True)
        for epoch in range(EPOCHS):
            model.train()
            lr = epoch_lr(epoch)
            for group in opt.param_groups:
                group["lr"] = lr
            total = 0.0
            correct = 0
            seen = 0
            for batch in train_loader:
                feats = batch["feat"].to(device)
                labels = batch["label"].to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(feats)
                loss = criterion(logits, labels)
                loss.backward()
                opt.step()
                total += float(loss.item()) * labels.shape[0]
                correct += int((logits.argmax(-1) == labels).sum().item())
                seen += int(labels.shape[0])
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch+1:02d}/{EPOCHS} loss {total/max(seen,1):.4f} "
                    f"acc {100*correct/max(seen,1):.1f}",
                    flush=True,
                )
        metrics, preds = evaluate(model, test_loader, device)
        payload = {"config": "videomae", "fold": fold, **metrics}
        (OUT_DIR / f"fold{fold}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (OUT_DIR / f"preds_fold{fold}.json").write_text(json.dumps(preds, indent=2) + "\n", encoding="utf-8")
        fold_rows.append(payload)
        print(
            f"fold {fold} test top1 {metrics['top1']:.2f} top5 {metrics['top5']:.2f} "
            f"mca {metrics['mca']:.2f} n {metrics['n']}",
            flush=True,
        )
    top1 = [row["top1"] for row in fold_rows]
    top5 = [row["top5"] for row in fold_rows]
    mca = [row["mca"] for row in fold_rows]
    summary = {
        "config": "videomae",
        "folds": list(range(5)),
        "top1_mean": float(np.mean(top1)),
        "top1_std": float(np.std(top1, ddof=1)),
        "top5_mean": float(np.mean(top5)),
        "top5_std": float(np.std(top5, ddof=1)),
        "mca_mean": float(np.mean(mca)),
        "mca_std": float(np.std(mca, ddof=1)),
        "per_fold": fold_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"videomae  top1 {summary['top1_mean']:.2f}±{summary['top1_std']:.2f}  "
        f"mca {summary['mca_mean']:.2f}±{summary['mca_std']:.2f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device(f"cuda:{args.gpu}")
    rows = read_protocol()
    if not args.train_only:
        extract_features(rows, device, args.batch_extract)
    if not args.extract_only:
        train_folds(rows, device)


if __name__ == "__main__":
    main()
