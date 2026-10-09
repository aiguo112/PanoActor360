#!/usr/bin/env python3
"""Zero-shot CLIP / SigLIP baselines on 360+x clips (T8).

Evaluates on the 4,259-clip protocol with the same 5 folds as Table I.
Views: ERP frame mean, cubemap max-over-faces, person-crop max, and
person+ERP logit average. Writes results/zeroshot.csv and per-fold preds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "outputs" / "protocol" / "segments_v1.csv"
CLASSES_PATH = REPO_ROOT / "outputs" / "protocol" / "classes_v1.json"
FRAMES = REPO_ROOT / "data" / "frames_2048"
META = REPO_ROOT / "data" / "person_meta"
TANGENT = REPO_ROOT / "data" / "person_tangent"
OUT_CSV = REPO_ROOT / "results" / "zeroshot.csv"
OUT_DIR = REPO_ROOT / "results" / "zeroshot"
N_FRAMES = 16
FACE_SIZE = 224

# Gerund / natural phrase for each protocol class.
PROMPT_NOUN = {
    "clapping": "clapping",
    "cooking": "cooking",
    "coughing": "coughing",
    "dancing": "dancing",
    "dressing": "getting dressed",
    "drinking": "drinking",
    "driving": "driving",
    "eating": "eating",
    "firework": "watching fireworks",
    "flashing": "using a camera flash",
    "housekeeping": "housekeeping",
    "hygiene practices": "personal hygiene",
    "laughing": "laughing",
    "moving things": "moving objects",
    "opening": "opening something",
    "operating computer": "using a computer",
    "operating phone": "using a phone",
    "photographing": "taking a photo",
    "playing": "playing",
    "playing instrument": "playing a musical instrument",
    "pouring": "pouring a liquid",
    "preparing food": "preparing food",
    "sitting": "sitting",
    "speaking": "speaking",
    "standing": "standing",
    "walking": "walking",
    "workout": "working out",
}
TEMPLATES = (
    "a photo of a person {action}",
    "a person is {action}",
    "{action}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clip", "siglip"],
        choices=("clip", "siglip"),
    )
    parser.add_argument("--max-persons", type=int, default=8)
    return parser.parse_args()


def read_protocol() -> list[dict]:
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
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
                    "class_name": classes[str(int(row["class_idx"]))],
                }
            )
    rows.sort(key=lambda row: int(row["seg_id"]))
    return rows


def load_model(name: str, device: torch.device):
    if name == "clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP", pretrained="webli"
        )
        tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP")
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.inference_mode()
def text_classifier(model, tokenizer, classes: list[str], device: torch.device) -> torch.Tensor:
    """Return L2-normalized class text embeddings [C, D] averaged over templates."""
    embeds = []
    for class_name in classes:
        action = PROMPT_NOUN.get(class_name, class_name)
        texts = [template.format(action=action) for template in TEMPLATES]
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1).mean(dim=0)
        embeds.append(F.normalize(feats, dim=0))
    return torch.stack(embeds, dim=0)


def erp_to_cubemap(bgr: np.ndarray) -> list[np.ndarray]:
    import py360convert

    cube = py360convert.e2c(bgr, face_w=FACE_SIZE, mode="bilinear", cube_format="list")
    return cube


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if size is not None:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def person_crop_paths(seg_id: str, max_persons: int) -> list[Path]:
    meta_path = META / f"{seg_id}.json"
    if not meta_path.is_file():
        return []
    document = json.loads(meta_path.read_text(encoding="utf-8"))
    ranked = []
    for frame_idx, people in enumerate(document.get("frames") or []):
        for person in people:
            path = TANGENT / seg_id / f"{frame_idx:02d}_{int(person['j'])}.jpg"
            if path.is_file():
                ranked.append((float(person.get("conf", 0.0)), path))
    ranked.sort(key=lambda item: -item[0])
    return [path for _conf, path in ranked[:max_persons]]


def preprocess_batch(paths: list[Path], preprocess, size: tuple[int, int] | None = None) -> torch.Tensor:
    images = []
    for path in paths:
        rgb = load_rgb(path, size=size)
        from PIL import Image

        images.append(preprocess(Image.fromarray(rgb)))
    return torch.stack(images, dim=0)


@torch.inference_mode()
def encode_images(model, images: torch.Tensor, device: torch.device, batch: int) -> torch.Tensor:
    outs = []
    for start in range(0, images.shape[0], batch):
        chunk = images[start : start + batch].to(device)
        feats = model.encode_image(chunk)
        outs.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(outs, dim=0)


def clip_logits(image_feats: torch.Tensor, text_feats: torch.Tensor) -> np.ndarray:
    return (image_feats @ text_feats.T).numpy()


def metrics(pred: np.ndarray, true: np.ndarray, n_classes: int) -> dict:
    top1 = 100.0 * float((pred == true).mean()) if len(true) else 0.0
    per = []
    for class_idx in range(n_classes):
        mask = true == class_idx
        if not mask.any():
            continue
        per.append(float((pred[mask] == true[mask]).mean()))
    mca = 100.0 * float(np.mean(per)) if per else 0.0
    return {"top1": top1, "mca": mca, "n": int(len(true))}


def summarize_folds(fold_scores: list[dict]) -> tuple[float, float, float, float]:
    top1 = [row["top1"] for row in fold_scores]
    mca = [row["mca"] for row in fold_scores]
    return (
        float(np.mean(top1)),
        float(np.std(top1, ddof=1)) if len(top1) > 1 else 0.0,
        float(np.mean(mca)),
        float(np.std(mca, ddof=1)) if len(mca) > 1 else 0.0,
    )


def evaluate_model(name: str, rows: list[dict], device: torch.device, batch: int, max_persons: int) -> list[dict]:
    print(f"loading {name}", flush=True)
    model, preprocess, tokenizer = load_model(name, device)
    class_names = [json.loads(CLASSES_PATH.read_text())[str(i)] for i in range(27)]
    text = text_classifier(model, tokenizer, class_names, device).cpu()

    # Accumulate logits per view.
    views = ("erp", "cubemap", "person", "person_erp")
    logits = {view: {} for view in views}

    for index, row in enumerate(rows, start=1):
        seg_id = row["seg_id"]
        frame_dir = FRAMES / seg_id
        frame_paths = [frame_dir / f"{f:02d}.jpg" for f in range(N_FRAMES)]
        if not all(path.is_file() for path in frame_paths):
            continue

        # (a) ERP mean over frames
        erp_imgs = preprocess_batch(frame_paths, preprocess, size=(448, 224))
        erp_feats = encode_images(model, erp_imgs, device, batch)
        erp_logit = clip_logits(erp_feats.mean(dim=0, keepdim=True), text)[0]
        logits["erp"][seg_id] = erp_logit

        # (b) cubemap: max over faces, mean over frames
        cube_logits = []
        for path in frame_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            faces = erp_to_cubemap(bgr)
            from PIL import Image

            face_t = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))) for face in faces])
            face_feats = encode_images(model, face_t, device, batch)
            face_scores = clip_logits(face_feats, text)
            cube_logits.append(face_scores.max(axis=0))
        logits["cubemap"][seg_id] = np.mean(np.stack(cube_logits, axis=0), axis=0)

        # (c) person crops: max over persons (fallback to ERP if none)
        crop_paths = person_crop_paths(seg_id, max_persons)
        if crop_paths:
            crop_t = preprocess_batch(crop_paths, preprocess)
            crop_feats = encode_images(model, crop_t, device, batch)
            person_logit = clip_logits(crop_feats, text).max(axis=0)
        else:
            person_logit = erp_logit
        logits["person"][seg_id] = person_logit
        logits["person_erp"][seg_id] = 0.5 * (person_logit + erp_logit)

        if index % 50 == 0 or index == len(rows):
            print(f"  {name}: {index}/{len(rows)} clips", flush=True)

    results = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for view in views:
        fold_scores = []
        for fold in range(5):
            fold_rows = [row for row in rows if row["fold"] == fold and row["seg_id"] in logits[view]]
            true = np.array([row["class_idx"] for row in fold_rows], dtype=np.int64)
            scores = np.stack([logits[view][row["seg_id"]] for row in fold_rows], axis=0)
            pred = scores.argmax(axis=1)
            stats = metrics(pred, true, 27)
            fold_scores.append(stats)
            pred_path = OUT_DIR / f"{name}_{view}_fold{fold}.json"
            payload = [
                {
                    "seg_id": row["seg_id"],
                    "true": int(row["class_idx"]),
                    "pred": int(pred[i]),
                    "top5": [int(x) for x in scores[i].argsort()[::-1][:5]],
                }
                for i, row in enumerate(fold_rows)
            ]
            pred_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        top1_m, top1_s, mca_m, mca_s = summarize_folds(fold_scores)
        results.append(
            {
                "model": name,
                "view": view,
                "top1_mean": top1_m,
                "top1_std": top1_s,
                "mca_mean": mca_m,
                "mca_std": mca_s,
                "n": sum(row["n"] for row in fold_scores),
            }
        )
        print(
            f"{name}/{view}: top1 {top1_m:.2f}±{top1_s:.2f}  mca {mca_m:.2f}±{mca_s:.2f}",
            flush=True,
        )
    return results


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device(f"cuda:{args.gpu}")
    rows = read_protocol()
    print(f"protocol clips with frames: {len(rows)}  device {device}", flush=True)
    all_rows = []
    for name in args.models:
        all_rows.extend(evaluate_model(name, rows, device, args.batch, args.max_persons))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "view", "top1_mean", "top1_std", "mca_mean", "mca_std", "n"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: (f"{value:.6f}" if isinstance(value, float) else value) for key, value in row.items()})
    print(f"wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
