#!/usr/bin/env python3
"""Train the 360+x action recognizer on cached DINOv2 features.

Valid segments are protocol rows that have feats/global/{seg_id}.pt. Configs
that use people also require feats/person/{seg_id}.pt (or feats/person_light
for full_light), and cubemap requires
feats/cubemap/{seg_id}.pt. Grid configs load feats/global8/{seg_id}.pt, written
by scripts/pool_global.py, into one fp16 CPU tensor.

Training augmentations are yaw roll (config flag), horizontal flip (p=0.5),
and frame dropout of a random 0-25% of frames. Person lat/lon in the feature
files are radians; a yaw of s columns adds s*pi/8 and wraps to [-pi, pi).

Each fold trains for 60 epochs and is scored at the final epoch. Summary
statistics are recomputed from the fold json files on disk, and
results/all_results.csv keeps one row per config. --tag writes that row and
the fold files under results/{config}_{tag}/. --tune holds out 15% of the
training videos and scores that validation split instead of the test fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from har.model import (  # noqa: E402
    FEAT_DIM,
    GRID_H,
    GRID_W,
    MAX_PERSONS,
    N_CLASSES,
    N_FRAMES,
    HARModel,
)

PROTOCOL_PATH = REPO_ROOT / "outputs" / "protocol" / "segments_v1.csv"
FEATS_ROOT = REPO_ROOT / "feats"
GLOBAL8_ROOT = FEATS_ROOT / "global8"
META_ROOT = REPO_ROOT / "data" / "person_meta"
RESULTS_ROOT = REPO_ROOT / "results"

EPOCHS = 60
WARMUP_EPOCHS = 5
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
SEED = 42
# One 8x16 column is 22.5 degrees == pi/8 radians.
YAW_STEP_RAD = math.pi / 8

# global_stream: cls | cubemap | grid | none
# temporal: mean | transformer (fusion models also use the 2-layer encoder)
# persons: None | tangent | naive
FULL = {
    "global_stream": "grid",
    "temporal": "transformer",
    "lat_enc": True,
    "lon_mode": "circular",
    "area_pool": True,
    "persons": "tangent",
    "mil": True,
    "yaw_roll": True,
}
CONFIGS = {
    "erp_cls_mean": {
        "global_stream": "cls",
        "temporal": "mean",
        "lat_enc": False,
        "lon_mode": "none",
        "area_pool": False,
        "persons": None,
        "mil": False,
        "yaw_roll": False,
    },
    "erp_cls_temporal": {
        "global_stream": "cls",
        "temporal": "transformer",
        "lat_enc": False,
        "lon_mode": "none",
        "area_pool": False,
        "persons": None,
        "mil": False,
        "yaw_roll": False,
    },
    "cubemap": {
        "global_stream": "cubemap",
        "temporal": "transformer",
        "lat_enc": False,
        "lon_mode": "none",
        "area_pool": False,
        "persons": None,
        "mil": False,
        "yaw_roll": False,
    },
    "grid_plain": {
        "global_stream": "grid",
        "temporal": "transformer",
        "lat_enc": False,
        "lon_mode": "none",
        "area_pool": False,
        "persons": None,
        "mil": False,
        "yaw_roll": False,
    },
    "grid_latlon": {
        "global_stream": "grid",
        "temporal": "transformer",
        "lat_enc": True,
        "lon_mode": "circular",
        "area_pool": True,
        "persons": None,
        "mil": False,
        "yaw_roll": False,
    },
    "full": dict(FULL),
    "full_light": {**FULL, "person_feats": "person_light"},
    "abl_naive": {**FULL, "persons": "naive"},
    "abl_no_lat": {**FULL, "lat_enc": False},
    "abl_lin_lon": {**FULL, "lon_mode": "linear"},
    "abl_no_area": {**FULL, "area_pool": False},
    "abl_no_mil": {**FULL, "mil": False},
    "abl_no_roll": {**FULL, "yaw_roll": False},
    "persons_only": {
        "global_stream": "none",
        "temporal": "transformer",
        "lat_enc": False,
        "lon_mode": "none",
        "area_pool": False,
        "persons": "tangent",
        "mil": True,
        "yaw_roll": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated test folds, default 0,1,2,3,4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--person_drop", type=float, default=0.0, help="Train-time probability of dropping each person token")
    parser.add_argument("--tag", default="", help="Suffix for the results directory and all_results.csv row")
    parser.add_argument("--tune", action="store_true", help="Score a 15% video holdout from the training folds")
    return parser.parse_args()


def parse_folds(text: str) -> list[int]:
    folds = [int(part) for part in text.split(",") if part.strip() != ""]
    if not folds or any(fold < 0 or fold > 4 for fold in folds):
        raise SystemExit("--folds must be a comma-separated subset of 0,1,2,3,4")
    return folds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wrap_pi(lon: np.ndarray) -> np.ndarray:
    return (lon + np.pi) % (2.0 * np.pi) - np.pi


def epoch_lr(epoch: int, epochs: int = EPOCHS) -> float:
    """Linear warmup over 5 epochs, then cosine over the remaining epochs."""
    if epoch < WARMUP_EPOCHS:
        return LR * float(epoch + 1) / float(WARMUP_EPOCHS)
    span = epochs - WARMUP_EPOCHS
    if span <= 0:
        return LR
    progress = (epoch - WARMUP_EPOCHS) / float(span)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_pt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_protocol(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "seg_id": str(int(row["seg_id"])),
                    "video_id": row["video_id"],
                    "class_idx": int(row["class_idx"]),
                    "fold": int(row["fold"]),
                }
            )
    return rows


def person_feat_folder(config: dict) -> str:
    return config.get("person_feats", "person")


def select_rows(rows: list[dict], config: dict) -> tuple[list[dict], dict]:
    kept = []
    missing = {"global": 0, "person": 0, "cubemap": 0}
    need_person = config["persons"] is not None
    need_cube = config["global_stream"] == "cubemap"
    person_dir = person_feat_folder(config)
    for row in rows:
        seg_id = row["seg_id"]
        if not (FEATS_ROOT / "global" / f"{seg_id}.pt").is_file():
            missing["global"] += 1
            continue
        if need_person and not (FEATS_ROOT / person_dir / f"{seg_id}.pt").is_file():
            missing["person"] += 1
            continue
        if need_cube and not (FEATS_ROOT / "cubemap" / f"{seg_id}.pt").is_file():
            missing["cubemap"] += 1
            continue
        kept.append(row)
    kept.sort(key=lambda row: int(row["seg_id"]))
    return kept, missing


def load_grid_tensor(needed: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
    """Load every feats/global8/{seg_id}.pt into one fp16 CPU tensor."""
    if not GLOBAL8_ROOT.is_dir():
        raise SystemExit(f"feats/global8 is missing ({GLOBAL8_ROOT}). Run scripts/pool_global.py first.")
    paths = sorted(GLOBAL8_ROOT.glob("*.pt"), key=lambda path: int(path.stem))
    present = {path.stem for path in paths}
    missing = [seg_id for seg_id in needed if seg_id not in present]
    if not paths or missing:
        preview = ", ".join(missing[:8])
        detail = f" Missing {len(missing)} of {len(needed)} required segments"
        if preview:
            detail += f", including {preview}"
        detail += "."
        if not paths:
            detail = " The directory has no pooled segment files."
        raise SystemExit(f"feats/global8 is incomplete ({GLOBAL8_ROOT}).{detail} Run scripts/pool_global.py.")
    expected = (N_FRAMES, GRID_H, GRID_W, FEAT_DIM)
    index = {path.stem: row for row, path in enumerate(paths)}
    grid = torch.empty((len(paths), *expected), dtype=torch.float16)
    print(f"loading {len(paths)} pooled grids from {GLOBAL8_ROOT}", flush=True)
    for path in paths:
        row = index[path.stem]
        tokens = load_pt(path)["tokens"].detach().cpu()
        if tuple(tokens.shape) != expected:
            raise SystemExit(
                f"feats/global8 is incomplete: {path.name} tokens have shape {tuple(tokens.shape)}, expected {expected}."
            )
        grid[row] = tokens.to(dtype=torch.float16)
        if (row + 1) % 200 == 0 or row + 1 == len(paths):
            print(f"  loaded grid {row + 1}/{len(paths)}", flush=True)
    gib = grid.numel() * grid.element_size() / (1024 ** 3)
    print(f"loaded global8 {tuple(grid.shape)} {grid.dtype} ({gib:.2f} GiB)", flush=True)
    return grid, index


def load_feature_stack(seg_ids: list[str], stream: str, tail: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    index = {seg_id: row for row, seg_id in enumerate(seg_ids)}
    stack = np.empty((len(seg_ids), *tail), dtype=np.float16)
    folder = "global8" if stream == "cls" else "cubemap"
    key = "cls"
    for seg_id, row in index.items():
        blob = load_pt(FEATS_ROOT / folder / f"{seg_id}.pt")
        value = blob[key].detach().cpu().numpy()
        if tuple(value.shape) != tail:
            raise RuntimeError(f"{seg_id} {stream} cls has shape {value.shape}")
        stack[row] = value
        if (row + 1) % 500 == 0 or row + 1 == len(seg_ids):
            print(f"  loaded {stream} {row + 1}/{len(seg_ids)}", flush=True)
    return stack, index


def load_boxes(seg_id: str) -> list[list[float]] | None:
    path = META_ROOT / f"{seg_id}.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
    for frame in document.get("frames") or []:
        for person in frame:
            boxes.append([float(person["x1"]), float(person["y1"]), float(person["x2"]), float(person["y2"])])
    return boxes


def numpy_1d(value, dtype) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def load_persons(rows: list[dict], kind: str, folder: str = "person") -> dict[str, dict]:
    packed = {}
    box_mismatch = 0
    for index, row in enumerate(rows):
        seg_id = row["seg_id"]
        blob = load_pt(FEATS_ROOT / folder / f"{seg_id}.pt")
        feat = numpy_1d(blob[kind], np.float16)
        frame_idx = numpy_1d(blob["frame_idx"], np.int64)
        lat = numpy_1d(blob["lat"], np.float32)
        lon = numpy_1d(blob["lon"], np.float32)
        fov = numpy_1d(blob["fov"], np.float32)
        conf = numpy_1d(blob["conf"], np.float32)
        count = int(feat.shape[0])
        if any(array.shape[0] != count for array in (frame_idx, lat, lon, fov, conf)):
            raise RuntimeError(f"{seg_id} person fields do not share a length")
        boxes = load_boxes(seg_id)
        if boxes is not None and len(boxes) == count:
            box = np.asarray(boxes, dtype=np.float32).reshape(count, 4)
            has_box = np.ones(count, dtype=bool)
        else:
            if count:
                box_mismatch += 1
            box = np.zeros((count, 4), dtype=np.float32)
            has_box = np.zeros(count, dtype=bool)
        if count > MAX_PERSONS:
            chosen = np.argsort(-conf, kind="stable")[:MAX_PERSONS]
            chosen.sort()
            feat, frame_idx = feat[chosen], frame_idx[chosen]
            lat, lon, fov, conf = lat[chosen], lon[chosen], fov[chosen], conf[chosen]
            box, has_box = box[chosen], has_box[chosen]
        packed[seg_id] = {
            "feat": feat,
            "frame_idx": frame_idx,
            "lat": lat,
            "lon": lon,
            "fov": fov,
            "conf": conf,
            "box": box,
            "has_box": has_box,
        }
        if (index + 1) % 500 == 0 or index + 1 == len(rows):
            print(f"  loaded persons {index + 1}/{len(rows)}", flush=True)
    if box_mismatch:
        print(f"person meta boxes missing or misaligned for {box_mismatch} segments", flush=True)
    return packed


class FeatureStore:
    def __init__(self, rows: list[dict], config: dict):
        self.config = config
        self.grid = None
        self.grid_index = {}
        self.frames = None
        self.frame_index = {}
        self.persons = {}
        seg_ids = [row["seg_id"] for row in rows]
        stream = config["global_stream"]
        if stream == "grid":
            self.grid, self.grid_index = load_grid_tensor(seg_ids)
        elif stream == "cls":
            print(f"loading global CLS for {len(seg_ids)} segments", flush=True)
            self.frames, self.frame_index = load_feature_stack(seg_ids, "cls", (N_FRAMES, FEAT_DIM))
        elif stream == "cubemap":
            print(f"loading cubemap CLS for {len(seg_ids)} segments", flush=True)
            self.frames, self.frame_index = load_feature_stack(seg_ids, "cubemap", (N_FRAMES, 6, FEAT_DIM))
        if config["persons"] is not None:
            folder = person_feat_folder(config)
            print(f"loading {config['persons']} person features from {folder} for {len(seg_ids)} segments", flush=True)
            self.persons = load_persons(rows, config["persons"], folder)


def holdout_videos(rows: list[dict], fraction: float = 0.15, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Hold out a fraction of videos. Every segment from a video stays in one split."""
    videos = sorted({row["video_id"] for row in rows})
    if len(videos) < 2:
        raise SystemExit("tune needs at least 2 training videos to hold out a validation set")
    order = list(videos)
    random.Random(seed).shuffle(order)
    n_val = int(round(len(order) * fraction))
    n_val = min(max(n_val, 1), len(order) - 1)
    val_ids = set(order[:n_val])
    train_rows = [row for row in rows if row["video_id"] not in val_ids]
    val_rows = [row for row in rows if row["video_id"] in val_ids]
    if not train_rows or not val_rows:
        raise SystemExit("tune video holdout produced an empty train or validation split")
    return train_rows, val_rows


def result_name(config: str, tag: str) -> str:
    return f"{config}_{tag}" if tag else config


def drop_person_tokens(persons: dict | None, prob: float) -> dict | None:
    if persons is None or prob <= 0.0:
        return persons
    count = int(persons["feat"].shape[0])
    if count == 0:
        return persons
    keep = np.fromiter((random.random() >= prob for _ in range(count)), dtype=bool, count=count)
    if bool(keep.all()):
        return persons
    return {key: value[keep] for key, value in persons.items()}


def augment(grid, persons, train: bool, yaw_roll: bool):
    frame_mask = np.ones(N_FRAMES, dtype=bool)
    if not train:
        return grid, persons, frame_mask
    if yaw_roll:
        shift = random.randrange(GRID_W)
        if grid is not None:
            grid = np.roll(grid, shift=shift, axis=2)
        if persons is not None and persons["lon"].size:
            persons["lon"] = wrap_pi(persons["lon"] + shift * YAW_STEP_RAD)
    if random.random() < 0.5:
        if grid is not None:
            grid = np.ascontiguousarray(grid[:, :, ::-1, :])
        if persons is not None and persons["lon"].size:
            persons["lon"] = wrap_pi(-persons["lon"])
    drop = random.randrange(N_FRAMES // 4 + 1)
    if drop:
        for frame in random.sample(range(N_FRAMES), drop):
            frame_mask[frame] = False
    return grid, persons, frame_mask


class SegmentDataset(Dataset):
    def __init__(self, rows: list[dict], store: FeatureStore, train: bool, person_drop: float = 0.0):
        self.rows = rows
        self.store = store
        self.train = train
        self.person_drop = person_drop
        self.config = store.config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        seg_id = row["seg_id"]
        stream = self.config["global_stream"]
        grid = None
        if stream == "grid":
            grid = self.store.grid[self.store.grid_index[seg_id]].float().numpy()
        persons = None
        if self.config["persons"] is not None:
            source = self.store.persons[seg_id]
            persons = {
                "feat": np.array(source["feat"], dtype=np.float32),
                "frame_idx": np.array(source["frame_idx"], dtype=np.int64),
                "lat": np.array(source["lat"], dtype=np.float32),
                "lon": np.array(source["lon"], dtype=np.float32),
                "fov": np.array(source["fov"], dtype=np.float32),
                "conf": np.array(source["conf"], dtype=np.float32),
                "box": np.array(source["box"], dtype=np.float32),
                "has_box": np.array(source["has_box"], dtype=bool),
            }
        grid, persons, frame_mask = augment(grid, persons, self.train, bool(self.config["yaw_roll"]))
        if self.train:
            persons = drop_person_tokens(persons, self.person_drop)
        sample = {
            "seg_id": seg_id,
            "label": int(row["class_idx"]),
            "frame_mask": torch.from_numpy(np.ascontiguousarray(frame_mask)),
        }
        if grid is not None:
            sample["grid"] = torch.from_numpy(np.ascontiguousarray(grid))
        if stream in ("cls", "cubemap"):
            frames = np.array(self.store.frames[self.store.frame_index[seg_id]], dtype=np.float32)
            key = "cls" if stream == "cls" else "faces"
            sample[key] = torch.from_numpy(np.ascontiguousarray(frames))
        if persons is not None:
            sample["person_feat"] = torch.from_numpy(np.ascontiguousarray(persons["feat"]))
            sample["person_frame"] = torch.from_numpy(np.ascontiguousarray(persons["frame_idx"]))
            sample["person_lat"] = torch.from_numpy(np.ascontiguousarray(persons["lat"]))
            sample["person_lon"] = torch.from_numpy(np.ascontiguousarray(persons["lon"]))
            sample["person_fov"] = torch.from_numpy(np.ascontiguousarray(persons["fov"]))
            sample["person_conf"] = torch.from_numpy(np.ascontiguousarray(persons["conf"]))
            sample["person_box"] = torch.from_numpy(np.ascontiguousarray(persons["box"]))
            sample["person_has_box"] = torch.from_numpy(np.ascontiguousarray(persons["has_box"]))
        return sample


def collate_segments(samples: list[dict]) -> dict:
    batch = {
        "seg_id": [sample["seg_id"] for sample in samples],
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "frame_mask": torch.stack([sample["frame_mask"] for sample in samples]),
    }
    for key in ("grid", "cls", "faces"):
        if key in samples[0]:
            batch[key] = torch.stack([sample[key] for sample in samples])
    if "person_feat" not in samples[0]:
        return batch
    batch_size = len(samples)
    feat = torch.zeros(batch_size, MAX_PERSONS, FEAT_DIM)
    frame = torch.zeros(batch_size, MAX_PERSONS, dtype=torch.long)
    lat = torch.zeros(batch_size, MAX_PERSONS)
    lon = torch.zeros(batch_size, MAX_PERSONS)
    fov = torch.zeros(batch_size, MAX_PERSONS)
    conf = torch.zeros(batch_size, MAX_PERSONS)
    box = torch.zeros(batch_size, MAX_PERSONS, 4)
    has_box = torch.zeros(batch_size, MAX_PERSONS, dtype=torch.bool)
    mask = torch.zeros(batch_size, MAX_PERSONS, dtype=torch.bool)
    person_count = torch.zeros(batch_size, dtype=torch.long)
    for index, sample in enumerate(samples):
        number = int(sample["person_feat"].shape[0])
        person_count[index] = number
        if number == 0:
            continue
        feat[index, :number] = sample["person_feat"]
        frame[index, :number] = sample["person_frame"]
        lat[index, :number] = sample["person_lat"]
        lon[index, :number] = sample["person_lon"]
        fov[index, :number] = sample["person_fov"]
        conf[index, :number] = sample["person_conf"]
        box[index, :number] = sample["person_box"]
        has_box[index, :number] = sample["person_has_box"]
        mask[index, :number] = True
    batch.update(
        {
            "person_feat": feat,
            "person_frame": frame,
            "person_lat": lat,
            "person_lon": lon,
            "person_fov": fov,
            "person_conf": conf,
            "person_box": box,
            "person_has_box": has_box,
            "person_mask": mask,
            "person_count": person_count,
        }
    )
    return batch


def class_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    counts = np.bincount([row["class_idx"] for row in rows], minlength=N_CLASSES).astype(np.float64)
    weights = np.zeros(N_CLASSES, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
            if value.is_floating_point():
                value = value.float()
            moved[key] = value
        else:
            moved[key] = value
    return moved


def make_loader(dataset: SegmentDataset, train: bool) -> DataLoader:
    generator = None
    if train:
        generator = torch.Generator()
        generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=0,
        collate_fn=collate_segments,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        number = int(labels.shape[0])
        total_loss += float(loss.item()) * number
        with torch.no_grad():
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
        seen += number
    return total_loss / max(seen, 1), 100.0 * correct / max(seen, 1)


@torch.inference_mode()
def evaluate(model, loader, device, save_alpha: bool):
    model.eval()
    logits_all = []
    labels_all = []
    seg_ids: list[str] = []
    records = []
    for batch in loader:
        moved = move_batch(batch, device)
        if save_alpha:
            logits, alpha = model(moved, return_alpha=True)
        else:
            logits = model(moved)
            alpha = None
        logits_all.append(logits.cpu())
        labels_all.append(moved["label"].cpu())
        seg_ids.extend(batch["seg_id"])
        if save_alpha:
            records.extend(alpha_records(batch, logits, alpha))
    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    return (
        metrics_from_logits(logits_cat, labels_cat),
        records,
        segment_predictions(seg_ids, logits_cat, labels_cat),
        logits_cat,
    )


def segment_predictions(seg_ids: list[str], logits: torch.Tensor, labels: torch.Tensor) -> list[dict]:
    if len(seg_ids) != int(logits.shape[0]):
        raise RuntimeError(f"prediction count {logits.shape[0]} != segment count {len(seg_ids)}")
    predicted = logits.argmax(dim=-1)
    top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    return [
        {
            "seg_id": str(seg_id),
            "true": int(labels[index]),
            "pred": int(predicted[index]),
            "top5": [int(class_idx) for class_idx in top5[index].tolist()],
        }
        for index, seg_id in enumerate(seg_ids)
    ]


def alpha_records(batch: dict, logits: torch.Tensor, alpha: torch.Tensor) -> list[dict]:
    predicted = logits.argmax(dim=-1).cpu()
    alpha_cpu = alpha.cpu()
    records = []
    counts = batch["person_count"].tolist()
    for index, seg_id in enumerate(batch["seg_id"]):
        number = int(counts[index])
        persons = []
        for slot in range(number):
            box = None
            if bool(batch["person_has_box"][index, slot]):
                box = [float(value) for value in batch["person_box"][index, slot].tolist()]
            persons.append(
                {
                    "frame_idx": int(batch["person_frame"][index, slot]),
                    "lat": float(batch["person_lat"][index, slot]),
                    "lon": float(batch["person_lon"][index, slot]),
                    "box": box,
                    "alpha": float(alpha_cpu[index, slot]),
                }
            )
        records.append(
            {
                "seg_id": seg_id,
                "predicted": int(predicted[index]),
                "true": int(batch["label"][index]),
                "null_alpha": float(alpha_cpu[index, MAX_PERSONS]),
                "persons": persons,
            }
        )
    return records


def write_prediction_npz(
    run_name: str,
    fold: int,
    rows: list[dict],
    preds: list[dict],
    logits: torch.Tensor,
    records: list[dict],
) -> None:
    """Per-clip dump: clip id, video id, label, logits, alpha, person metadata."""
    video_of = {row["seg_id"]: row["video_id"] for row in rows}
    alpha_of = {str(record["seg_id"]): record for record in records}
    clip_id = np.array([str(item["seg_id"]) for item in preds])
    video_id = np.array([video_of.get(str(item["seg_id"]), "") for item in preds])
    label = np.array([int(item["true"]) for item in preds], dtype=np.int64)
    pred = np.array([int(item["pred"]) for item in preds], dtype=np.int64)
    if int(logits.shape[0]) != len(preds):
        raise RuntimeError(f"logit rows {logits.shape[0]} != predictions {len(preds)}")
    alpha = np.empty(len(preds), dtype=object)
    person_meta = np.empty(len(preds), dtype=object)
    for index, item in enumerate(preds):
        record = alpha_of.get(str(item["seg_id"]))
        if record is None:
            alpha[index] = np.zeros(0, dtype=np.float32)
            person_meta[index] = []
        else:
            alpha[index] = np.array(
                [float(person["alpha"]) for person in record["persons"]] + [float(record["null_alpha"])],
                dtype=np.float32,
            )
            person_meta[index] = record["persons"]
    directory = RESULTS_ROOT / "preds"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}_fold{fold}.npz"
    temporary = directory / f".{run_name}_fold{fold}.npz.partial"
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            clip_id=clip_id,
            video_id=video_id,
            label=label,
            pred=pred,
            logits=logits.numpy().astype(np.float32),
            alpha=alpha,
            person_meta=person_meta,
        )
    os.replace(temporary, path)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    predicted = logits.argmax(dim=-1)
    top1 = float((predicted == labels).float().mean().item() * 100.0)
    topk = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    top5 = float((topk == labels[:, None]).any(dim=-1).float().mean().item() * 100.0)
    per_class = {}
    class_scores = []
    for class_idx in sorted(int(value) for value in labels.unique().tolist()):
        chosen = labels == class_idx
        accuracy = float((predicted[chosen] == labels[chosen]).float().mean().item() * 100.0)
        per_class[str(class_idx)] = accuracy
        class_scores.append(accuracy)
    mca = float(sum(class_scores) / len(class_scores)) if class_scores else 0.0
    return {"top1": top1, "top5": top5, "mca": mca, "n": int(labels.shape[0]), "per_class_acc": per_class}


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return mean, std


def refresh_summary(config_name: str) -> dict:
    directory = RESULTS_ROOT / config_name
    folds = []
    for path in sorted(directory.glob("fold*.json"), key=lambda item: int(item.stem.replace("fold", ""))):
        folds.append(json.loads(path.read_text(encoding="utf-8")))
    if not folds:
        raise RuntimeError(f"no fold results in {directory}")
    top1_mean, top1_std = mean_std([fold["top1"] for fold in folds])
    top5_mean, top5_std = mean_std([fold["top5"] for fold in folds])
    mca_mean, mca_std = mean_std([fold["mca"] for fold in folds])
    summary = {
        "config": config_name,
        "folds": [fold["fold"] for fold in folds],
        "top1_mean": top1_mean,
        "top1_std": top1_std,
        "top5_mean": top5_mean,
        "top5_std": top5_std,
        "mca_mean": mca_mean,
        "mca_std": mca_std,
        "per_fold": [
            {"fold": fold["fold"], "top1": fold["top1"], "top5": fold["top5"], "mca": fold["mca"], "n": fold["n"]}
            for fold in folds
        ],
    }
    write_json(directory / "summary.json", summary)
    update_results_csv(summary)
    return summary


def update_results_csv(summary: dict) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULTS_ROOT / "all_results.csv"
    fieldnames = [
        "config",
        "top1_mean",
        "top1_std",
        "top5_mean",
        "top5_std",
        "mca_mean",
        "mca_std",
    ]
    rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("config") != summary["config"]]
    rows.append(
        {
            "config": summary["config"],
            "top1_mean": f"{summary['top1_mean']:.6f}",
            "top1_std": f"{summary['top1_std']:.6f}",
            "top5_mean": f"{summary['top5_mean']:.6f}",
            "top5_std": f"{summary['top5_std']:.6f}",
            "mca_mean": f"{summary['mca_mean']:.6f}",
            "mca_std": f"{summary['mca_std']:.6f}",
        }
    )
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_fold(
    fold: int,
    rows: list[dict],
    store: FeatureStore,
    config: dict,
    config_name: str,
    device,
    dropout: float = 0.1,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int = EPOCHS,
    person_drop: float = 0.0,
    run_name: str | None = None,
    tag: str = "",
    tune: bool = False,
) -> dict | None:
    run_name = config_name if run_name is None else run_name
    set_seed(SEED)
    pool_rows = [row for row in rows if row["fold"] != fold]
    test_rows = [row for row in rows if row["fold"] == fold]
    if not pool_rows or not test_rows:
        raise SystemExit(f"fold {fold} has {len(pool_rows)} train and {len(test_rows)} test segments")
    if tune:
        train_rows, eval_rows = holdout_videos(pool_rows)
    else:
        train_rows = pool_rows
        eval_rows = test_rows
    model = HARModel(config, dropout=dropout).to(device)
    random.seed(SEED)
    np.random.seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=epoch_lr(0, epochs), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device), label_smoothing=LABEL_SMOOTHING)
    train_loader = make_loader(SegmentDataset(train_rows, store, train=True, person_drop=person_drop), train=True)
    eval_loader = make_loader(SegmentDataset(eval_rows, store, train=False), train=False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if tune:
        n_train_videos = len({row["video_id"] for row in train_rows})
        n_val_videos = len({row["video_id"] for row in eval_rows})
        print(
            f"fold {fold}  train {len(train_rows)} ({n_train_videos} videos)  "
            f"val {len(eval_rows)} ({n_val_videos} videos)  parameters {n_params}",
            flush=True,
        )
    else:
        print(
            f"fold {fold}  train {len(train_rows)}  test {len(test_rows)}  parameters {n_params}",
            flush=True,
        )
    for epoch in range(epochs):
        lr = epoch_lr(epoch, epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(
            f"fold {fold} epoch {epoch + 1:02d}/{epochs} loss {loss:.4f} train_acc {train_acc:.2f} lr {lr:.3e}",
            flush=True,
        )
    save_alpha = (not tune) and bool(config.get("mil"))
    metrics, records, preds, logits = evaluate(model, eval_loader, device, save_alpha=save_alpha)
    if tune:
        payload = {
            "config": config_name,
            "tag": tag,
            "fold": fold,
            "split": "val",
            "top1": metrics["top1"],
            "mca": metrics["mca"],
            "n": metrics["n"],
            "n_train": len(train_rows),
            "n_train_videos": len({row["video_id"] for row in train_rows}),
            "n_val_videos": len({row["video_id"] for row in eval_rows}),
        }
        write_json(RESULTS_ROOT / "tune" / f"{run_name}_fold{fold}.json", payload)
        print(
            f"tune fold {fold} val top1 {metrics['top1']:.2f} mca {metrics['mca']:.2f} n {metrics['n']}",
            flush=True,
        )
        return {"fold": fold, "top1": metrics["top1"], "mca": metrics["mca"]}
    payload = {
        "config": config_name,
        "fold": fold,
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "mca": metrics["mca"],
        "n": metrics["n"],
        "n_train": len(train_rows),
        "per_class_acc": metrics["per_class_acc"],
    }
    write_json(RESULTS_ROOT / run_name / f"fold{fold}.json", payload)
    write_json(RESULTS_ROOT / run_name / f"preds_fold{fold}.json", preds)
    if records:
        write_json(
            RESULTS_ROOT / run_name / f"alpha_fold{fold}.json",
            {
                "config": config_name,
                "fold": fold,
                "lat_lon_unit": "radian",
                "box": "erp_xyxy",
                "segments": records,
            },
        )
    write_prediction_npz(run_name, fold, rows, preds, logits, records)
    print(
        f"fold {fold} test top1 {metrics['top1']:.2f} top5 {metrics['top5']:.2f} "
        f"mca {metrics['mca']:.2f} n {metrics['n']}",
        flush=True,
    )
    return None


def check_train_args(args: argparse.Namespace) -> None:
    if args.dropout < 0.0:
        raise SystemExit("--dropout must be >= 0")
    if args.wd < 0.0:
        raise SystemExit("--wd must be >= 0")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if not 0.0 <= args.person_drop <= 1.0:
        raise SystemExit("--person_drop must be between 0 and 1")


def main() -> None:
    args = parse_args()
    check_train_args(args)
    folds = parse_folds(args.folds)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"--gpu {args.gpu} is outside 0..{torch.cuda.device_count() - 1}")
    device = torch.device(f"cuda:{args.gpu}")
    config = CONFIGS[args.config]
    name = result_name(args.config, args.tag)
    rows, missing = select_rows(read_protocol(PROTOCOL_PATH), config)
    print(
        f"config {name}  device {device}  segments {len(rows)}  "
        f"skipped global {missing['global']} person {missing['person']} cubemap {missing['cubemap']}",
        flush=True,
    )
    if not rows:
        raise SystemExit("no valid segments for this config")
    store = FeatureStore(rows, config)
    tuned = []
    for fold in folds:
        metrics = run_fold(
            fold,
            rows,
            store,
            config,
            args.config,
            device,
            dropout=args.dropout,
            weight_decay=args.wd,
            epochs=args.epochs,
            person_drop=args.person_drop,
            run_name=name,
            tag=args.tag,
            tune=args.tune,
        )
        if args.tune:
            tuned.append(metrics)
            continue
        summary = refresh_summary(name)
        print(
            f"{name}  folds {summary['folds']}  "
            f"top1 {summary['top1_mean']:.2f} +/- {summary['top1_std']:.2f}  "
            f"top5 {summary['top5_mean']:.2f} +/- {summary['top5_std']:.2f}  "
            f"mca {summary['mca_mean']:.2f} +/- {summary['mca_std']:.2f}",
            flush=True,
        )
    if args.tune and tuned:
        top1_mean, top1_std = mean_std([item["top1"] for item in tuned])
        mca_mean, mca_std = mean_std([item["mca"] for item in tuned])
        print(
            f"tune {name}  top1 {top1_mean:.2f} +/- {top1_std:.2f}  mca {mca_mean:.2f} +/- {mca_std:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
