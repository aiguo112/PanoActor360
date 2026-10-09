#!/usr/bin/env python3
"""Detect people on ERP frames and write tangent plus naive crops.

Reads outputs/protocol/segments_v1.csv and data/frames_2048/{seg_id}/{k:02d}.jpg.
Detection runs on GPU 1 (CUDA_VISIBLE_DEVICES=1). Crops are written by a CPU
pool. A segment is skipped when data/person_meta{suffix}/{seg_id}.json already exists.
Does not modify existing dataset or protocol files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import random
import sys
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Physical GPU 1. Must be set before torch / ultralytics are imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "outputs" / "protocol" / "segments_v1.csv"
DEFAULT_FRAMES = REPO_ROOT / "data" / "frames_2048"
DEFAULT_TANGENT = REPO_ROOT / "data" / "person_tangent"
DEFAULT_NAIVE = REPO_ROOT / "data" / "person_naive"
DEFAULT_META = REPO_ROOT / "data" / "person_meta"
DEFAULT_DEBUG = REPO_ROOT / "outputs" / "debug_crops"

ERP_W = 2048
ERP_H = 1024
PAD = 256
N_FRAMES = 16
TOP_K = 8
CONF_MIN = 0.3
NMS_IOU = 0.5
IMGSZ = 2560
OUT_SIZE = 224
FOV_MIN = 20.0
FOV_MAX = 90.0
FOV_SCALE = 1.3
N_WORKERS = 16
BATCH_SIZE = 4
PREDICT_BATCH = 8
MAX_CROP_JOBS = 64
PREFETCH_THREADS = 8
DEBUG_COUNT = 20
PERSON_CLASS = 0
LAT_BIN_DEG = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--tangent", type=Path, default=DEFAULT_TANGENT)
    parser.add_argument("--naive", type=Path, default=DEFAULT_NAIVE)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N pending segments")
    parser.add_argument("--debug-count", type=int, default=DEBUG_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--suffix", default="", help="Appended to person_meta, person_tangent, person_naive, and debug_crops")
    args = parser.parse_args()
    if args.meta == DEFAULT_META:
        args.meta = REPO_ROOT / "data" / f"person_meta{args.suffix}"
    if args.tangent == DEFAULT_TANGENT:
        args.tangent = REPO_ROOT / "data" / f"person_tangent{args.suffix}"
    if args.naive == DEFAULT_NAIVE:
        args.naive = REPO_ROOT / "data" / f"person_naive{args.suffix}"
    if args.debug == DEFAULT_DEBUG:
        args.debug = REPO_ROOT / "outputs" / f"debug_crops{args.suffix}"
    return args


def load_segment_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No segments in {path}")
    return [row["seg_id"] for row in rows]


def frame_path(frames_root: Path, seg_id: str, k: int) -> Path:
    return frames_root / str(seg_id) / f"{k:02d}.jpg"


def meta_path(meta_root: Path, seg_id: str) -> Path:
    return meta_root / f"{seg_id}.json"


def wrap_pad(image: np.ndarray, pad: int = PAD) -> np.ndarray:
    """Pad ERP horizontally so the 0/width seam is interior. Result is 2560x1024."""
    left = image[:, -pad:]
    right = image[:, :pad]
    return np.concatenate([left, image, right], axis=1)


def load_erp(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[0] != ERP_H or image.shape[1] != ERP_W:
        image = cv2.resize(image, (ERP_W, ERP_H), interpolation=cv2.INTER_AREA)
    return image


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def wrapped_iou(a, b, width: float = ERP_W) -> float:
    best = 0.0
    for shift in (-width, 0.0, width):
        shifted = (b[0] + shift, b[1], b[2] + shift, b[3])
        best = max(best, box_iou(a, shifted))
    return best


def canonicalize(x1: float, y1: float, x2: float, y2: float):
    """Put the box center into [0, ERP_W). Seam boxes keep x1<0 or x2>ERP_W."""
    if x2 <= x1 or y2 <= y1:
        return None
    cx = 0.5 * (x1 + x2)
    shift = math.floor(cx / ERP_W) * ERP_W
    x1 -= shift
    x2 -= shift
    y1 = float(min(max(y1, 0.0), ERP_H))
    y2 = float(min(max(y2, 0.0), ERP_H))
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    return x1, y1, x2, y2


def nms_wrapped(boxes: list[tuple], iou_thresh: float = NMS_IOU) -> list[tuple]:
    """Greedy NMS. boxes are (x1, y1, x2, y2, conf) with centers modulo ERP_W."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][4], reverse=True)
    suppressed = [False] * len(boxes)
    kept = []
    for index in order:
        if suppressed[index]:
            continue
        kept.append(boxes[index])
        ax1, ay1, ax2, ay2, _conf = boxes[index]
        for other in order:
            if suppressed[other] or other == index:
                continue
            bx1, by1, bx2, by2, _other_conf = boxes[other]
            if wrapped_iou((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)) >= iou_thresh:
                suppressed[other] = True
    return kept


def detections_from_result(result) -> list[dict]:
    """Shift padded-image boxes back by -PAD, seam-dedup, and keep the top 8."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    raw = []
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        if float(conf) < CONF_MIN:
            continue
        canon = canonicalize(float(x1) - PAD, float(y1), float(x2) - PAD, float(y2))
        if canon is None:
            continue
        raw.append((*canon, float(conf)))
    deduped = nms_wrapped(raw)[:TOP_K]
    people = []
    for j, (x1, y1, x2, y2, conf) in enumerate(deduped):
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        lon = (cx / ERP_W) * (2.0 * math.pi) - math.pi
        lat = (math.pi / 2.0) - (cy / ERP_H) * math.pi
        w_ang = ((x2 - x1) / ERP_W) * 360.0
        h_ang = ((y2 - y1) / ERP_H) * 180.0
        fov = min(FOV_MAX, max(FOV_MIN, FOV_SCALE * max(w_ang, h_ang)))
        seam = 1 if (x1 < 0.0 or x2 > ERP_W) else 0
        people.append(
            {
                "j": j,
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "conf": round(conf, 4),
                "lon": lon,
                "lat": lat,
                "fov": round(fov, 2),
                "seam": seam,
            }
        )
    return people


def tangent_maps(lon: float, lat: float, fov_deg: float, out_size: int = OUT_SIZE):
    """Gnomonic maps. +u is east (increasing lon), +v is down (decreasing lat)."""
    fov = math.radians(float(fov_deg))
    focal = (out_size / 2.0) / math.tan(fov / 2.0)
    vs = np.arange(out_size, dtype=np.float64) + 0.5
    us = np.arange(out_size, dtype=np.float64) + 0.5
    nx = (us[None, :] - out_size / 2.0) / focal
    ny = (out_size / 2.0 - vs[:, None]) / focal
    nz = np.ones((out_size, out_size), dtype=np.float64)

    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)
    cos_lat = math.cos(lat)
    sin_lat = math.sin(lat)
    right = np.array([cos_lon, 0.0, -sin_lon], dtype=np.float64)
    up = np.array([-sin_lat * sin_lon, cos_lat, -sin_lat * cos_lon], dtype=np.float64)
    forward = np.array([cos_lat * sin_lon, sin_lat, cos_lat * cos_lon], dtype=np.float64)

    dx = nx * right[0] + ny * up[0] + nz * forward[0]
    dy = nx * right[1] + ny * up[1] + nz * forward[1]
    dz = nx * right[2] + ny * up[2] + nz * forward[2]
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= norm
    dy /= norm
    dz /= norm

    lon_s = np.arctan2(dx, dz)
    lat_s = np.arcsin(np.clip(dy, -1.0, 1.0))
    map_x = np.mod((lon_s + math.pi) / (2.0 * math.pi) * ERP_W, ERP_W).astype(np.float32)
    map_y = ((math.pi / 2.0 - lat_s) / math.pi * ERP_H).astype(np.float32)
    return map_x, map_y


def render_tangent(image: np.ndarray, lon: float, lat: float, fov_deg: float) -> np.ndarray:
    """Rectilinear crop. Source is wrapped a few columns so bilinear samples cross the seam."""
    wrap = 2
    padded = np.concatenate([image[:, -wrap:], image, image[:, :wrap]], axis=1)
    map_x, map_y = tangent_maps(lon, lat, fov_deg)
    map_x = map_x + wrap
    map_y = np.clip(map_y, 0.0, ERP_H - 1.0)
    return cv2.remap(
        padded,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _hcat(parts: list[np.ndarray]) -> np.ndarray:
    usable = [part for part in parts if part is not None and part.size and part.shape[1] > 0 and part.shape[0] > 0]
    if not usable:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return np.concatenate(usable, axis=1)


def render_naive(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Axis-aligned ERP crop. Seam boxes are concatenated, then padded to a square."""
    height, width = image.shape[:2]
    y1i = max(0, min(height, int(math.floor(y1))))
    y2i = max(0, min(height, int(math.ceil(y2))))
    if y2i <= y1i:
        y1i = max(0, min(height - 1, y1i))
        y2i = min(height, y1i + 1)
    span = x2 - x1
    if span >= width:
        strip = image[y1i:y2i, :]
    elif x1 < 0.0:
        left = image[y1i:y2i, max(0, int(math.floor(x1 + width))) : width]
        right = image[y1i:y2i, 0 : max(0, min(width, int(math.ceil(x2))))]
        strip = _hcat([left, right])
    elif x2 > width:
        left = image[y1i:y2i, max(0, min(width, int(math.floor(x1)))) : width]
        right = image[y1i:y2i, 0 : max(0, min(width, int(math.ceil(x2 - width))))]
        strip = _hcat([left, right])
    else:
        x1i = max(0, min(width - 1, int(math.floor(x1))))
        x2i = max(x1i + 1, min(width, int(math.ceil(x2))))
        strip = image[y1i:y2i, x1i:x2i]
    crop_h, crop_w = strip.shape[:2]
    side = max(crop_h, crop_w, 1)
    canvas = np.zeros((side, side, 3), dtype=strip.dtype)
    y0 = (side - crop_h) // 2
    x0 = (side - crop_w) // 2
    canvas[y0 : y0 + crop_h, x0 : x0 + crop_w] = strip
    return cv2.resize(canvas, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


def _init_worker() -> None:
    cv2.setNumThreads(1)


def crop_segment(job: dict) -> dict:
    """Write tangent and naive crops, then the segment meta JSON."""
    seg_id = job["seg_id"]
    frames = job["frames"]
    tangent_root = Path(job["tangent_root"])
    naive_root = Path(job["naive_root"])
    meta_file = Path(job["meta_file"])
    saved = []
    payload_frames = []
    try:
        for frame in frames:
            k = int(frame["k"])
            people = frame["people"]
            image = load_erp(Path(frame["path"])) if people else None
            if people and image is None:
                raise RuntimeError(f"failed to read {frame['path']}")
            tangent_dir = tangent_root / str(seg_id)
            naive_dir = naive_root / str(seg_id)
            if people:
                tangent_dir.mkdir(parents=True, exist_ok=True)
                naive_dir.mkdir(parents=True, exist_ok=True)
            for person in people:
                j = int(person["j"])
                tangent = render_tangent(image, person["lon"], person["lat"], person["fov"])
                naive = render_naive(image, person["x1"], person["y1"], person["x2"], person["y2"])
                tangent_path = tangent_dir / f"{k:02d}_{j}.jpg"
                naive_path = naive_dir / f"{k:02d}_{j}.jpg"
                if not cv2.imwrite(str(tangent_path), tangent):
                    raise RuntimeError(f"failed to write {tangent_path}")
                if not cv2.imwrite(str(naive_path), naive):
                    raise RuntimeError(f"failed to write {naive_path}")
                saved.append({"seg_id": seg_id, "k": k, "j": j})
            payload_frames.append(people)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        document = {"seg_id": seg_id, "frames": payload_frames}
        temporary = meta_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document), encoding="utf-8")
        os.replace(temporary, meta_file)
    except Exception as exc:
        return {"seg_id": seg_id, "ok": False, "error": str(exc), "saved": []}
    return {"seg_id": seg_id, "ok": True, "error": "", "saved": saved}


def draw_erp_box(image: np.ndarray, person: dict) -> np.ndarray:
    canvas = image.copy()
    x1 = float(person["x1"])
    y1 = float(person["y1"])
    x2 = float(person["x2"])
    y2 = float(person["y2"])
    color = (0, 255, 255) if int(person["seam"]) else (0, 255, 0)
    height, width = canvas.shape[:2]

    def rect(xa, ya, xb, yb):
        pt1 = (int(round(xa)), int(round(ya)))
        pt2 = (int(round(xb)), int(round(yb)))
        cv2.rectangle(canvas, pt1, pt2, color, 3)

    if x1 < 0.0:
        rect(0, y1, x2, y2)
        rect(x1 + width, y1, width - 1, y2)
    elif x2 > width:
        rect(x1, y1, width - 1, y2)
        rect(0, y1, x2 - width, y2)
    else:
        rect(x1, y1, x2, y2)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    cv2.circle(canvas, (int(round(cx % width)), int(round(min(max(cy, 0.0), height - 1)))), 8, (0, 0, 255), -1)
    return canvas


def write_debug_image(frames_root: Path, tangent_root: Path, naive_root: Path, out_dir: Path, item: dict) -> None:
    seg_id = item["seg_id"]
    k = int(item["k"])
    j = int(item["j"])
    person = item["person"]
    erp = load_erp(frame_path(frames_root, seg_id, k))
    tangent = cv2.imread(str(tangent_root / str(seg_id) / f"{k:02d}_{j}.jpg"), cv2.IMREAD_COLOR)
    naive = cv2.imread(str(naive_root / str(seg_id) / f"{k:02d}_{j}.jpg"), cv2.IMREAD_COLOR)
    if erp is None or tangent is None or naive is None:
        return
    boxed = draw_erp_box(erp, person)
    panel_h = 448
    erp_panel = cv2.resize(boxed, (panel_h * 2, panel_h), interpolation=cv2.INTER_AREA)
    tangent_panel = cv2.resize(tangent, (panel_h, panel_h), interpolation=cv2.INTER_NEAREST)
    naive_panel = cv2.resize(naive, (panel_h, panel_h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(erp_panel, "ERP", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(tangent_panel, "tangent", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(naive_panel, "naive", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    gap = np.full((panel_h, 8, 3), 255, dtype=np.uint8)
    body = np.concatenate([erp_panel, gap, tangent_panel, gap, naive_panel], axis=1)
    header = np.zeros((48, body.shape[1], 3), dtype=np.uint8)
    lon_deg = math.degrees(float(person["lon"]))
    lat_deg = math.degrees(float(person["lat"]))
    label = (
        f"seg {seg_id}  k={k:02d} j={j}  "
        f"lon={lon_deg:.1f} lat={lat_deg:.1f}  fov={float(person['fov']):.1f}  seam={int(person['seam'])}"
    )
    cv2.putText(header, label, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.concatenate([header, body], axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{seg_id}_{k:02d}_{j}.jpg"), canvas)


def iter_people(document: dict):
    frames = document.get("frames") or []
    for k, people in enumerate(frames):
        for person in people:
            yield k, person


def print_stats(meta_root: Path, segment_ids: list[str]) -> None:
    frame_counts = Counter()
    lat_bins = [0] * (180 // LAT_BIN_DEG)
    total_people = 0
    seam_people = 0
    frames_seen = 0
    empty_frames = 0
    segments_seen = 0
    empty_segments = 0
    missing_meta = 0
    for seg_id in segment_ids:
        path = meta_path(meta_root, seg_id)
        if not path.is_file():
            missing_meta += 1
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        frames = document.get("frames") or []
        segments_seen += 1
        if frames and all(len(people) == 0 for people in frames):
            empty_segments += 1
        for people in frames:
            n = len(people)
            frame_counts[n] += 1
            frames_seen += 1
            if n == 0:
                empty_frames += 1
            for person in people:
                total_people += 1
                if int(person.get("seam", 0)):
                    seam_people += 1
                lat_deg = math.degrees(float(person["lat"]))
                bin_index = int((lat_deg + 90.0) // LAT_BIN_DEG)
                if lat_deg >= 90.0:
                    bin_index = len(lat_bins) - 1
                bin_index = min(max(bin_index, 0), len(lat_bins) - 1)
                lat_bins[bin_index] += 1

    print(f"Total persons: {total_people}", flush=True)
    print("Persons per frame:", flush=True)
    max_count = max(frame_counts) if frame_counts else 0
    for count in range(max_count + 1):
        print(f"  {count:2d}: {frame_counts[count]}", flush=True)
    if frames_seen:
        print(f"Frames with 0 persons: {100.0 * empty_frames / frames_seen:.2f}% ({empty_frames}/{frames_seen})", flush=True)
    else:
        print("Frames with 0 persons: n/a", flush=True)
    if segments_seen:
        print(
            "Segments with 0 persons in all 16 frames: "
            f"{100.0 * empty_segments / segments_seen:.2f}% ({empty_segments}/{segments_seen})",
            flush=True,
        )
    else:
        print("Segments with 0 persons in all 16 frames: n/a", flush=True)
    print("Detection latitudes, 15-degree bins:", flush=True)
    for index, count in enumerate(lat_bins):
        lo = -90 + index * LAT_BIN_DEG
        hi = lo + LAT_BIN_DEG
        bracket = "]" if index == len(lat_bins) - 1 else ")"
        print(f"  [{lo:4d}, {hi:4d}{bracket}: {count}", flush=True)
    if total_people:
        print(f"Seam-crossing boxes: {seam_people} ({100.0 * seam_people / total_people:.2f}%)", flush=True)
    else:
        print("Seam-crossing boxes: 0 (n/a)", flush=True)
    if missing_meta:
        print(f"Segments without meta JSON: {missing_meta}", flush=True)


def format_duration(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def submit_frame_loads(executor: ThreadPoolExecutor, frames_root: Path, seg_id: str):
    """Queue 16 JPEG reads. None means the segment does not have 16 frame files."""
    paths = [frame_path(frames_root, seg_id, k) for k in range(N_FRAMES)]
    if not all(path.is_file() for path in paths):
        return None
    return [executor.submit(load_erp, path) for path in paths]


def collect_frame_loads(futures) -> list[np.ndarray] | None:
    if futures is None:
        return None
    images = [future.result() for future in futures]
    if any(image is None for image in images):
        return None
    return images


def meta_ids_on_disk(meta_root: Path) -> list[str]:
    if not meta_root.is_dir():
        return []

    def sort_key(stem: str):
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted((path.stem for path in meta_root.glob("*.json")), key=sort_key)


def main() -> None:
    args = parse_args()
    segment_ids = load_segment_ids(args.csv)
    pending_all = [seg_id for seg_id in segment_ids if not meta_path(args.meta, seg_id).is_file()]
    pending = pending_all[: args.limit] if args.limit is not None else pending_all
    already = len(segment_ids) - len(pending_all)
    message = f"Segments: {len(segment_ids)} total, {already} already have meta, {len(pending_all)} pending"
    if args.limit is not None:
        message += f", processing {len(pending)}"
    print(message, flush=True)

    # Spawn the crop pool before torch so workers do not inherit a CUDA context.
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(processes=args.workers, initializer=_init_worker) if pending else None
    executor = ThreadPoolExecutor(max_workers=PREFETCH_THREADS) if pending else None
    inflight = deque()
    counts = {"ok": 0, "failed": 0, "crops": 0}

    def reap_oldest() -> None:
        seg_id, async_result = inflight.popleft()
        try:
            item = async_result.get()
        except Exception:
            counts["failed"] += 1
            print(f"crop failed seg {seg_id}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            return
        if not item["ok"]:
            counts["failed"] += 1
            print(f"crop failed seg {item['seg_id']}: {item['error']}", file=sys.stderr, flush=True)
            return
        counts["ok"] += 1
        counts["crops"] += len(item["saved"])

    try:
        if pending:
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise SystemExit("CUDA is not available with CUDA_VISIBLE_DEVICES=1")
            cv2.setNumThreads(1)
            model = YOLO(args.model)
            started = time.perf_counter()
            frames_done = 0
            next_load = submit_frame_loads(executor, args.frames, pending[0])
            for index, seg_id in enumerate(pending):
                images = collect_frame_loads(next_load)
                if index + 1 < len(pending):
                    next_load = submit_frame_loads(executor, args.frames, pending[index + 1])
                if images is None:
                    print(
                        f"Skipping {seg_id}: fewer than {N_FRAMES} readable frames; meta not written",
                        flush=True,
                    )
                else:
                    people = []
                    for start in range(0, N_FRAMES, PREDICT_BATCH):
                        chunk = images[start : start + PREDICT_BATCH]
                        padded = [wrap_pad(image) for image in chunk]
                        results = model.predict(
                            source=padded,
                            imgsz=args.imgsz,
                            conf=CONF_MIN,
                            classes=[PERSON_CLASS],
                            device=0,
                            rect=True,
                            verbose=False,
                            save=False,
                            stream=False,
                            batch=len(padded),
                            half=True,
                        )
                        if len(results) != len(padded):
                            raise RuntimeError(f"expected {len(padded)} detections, got {len(results)}")
                        for result in results:
                            people.append(detections_from_result(result))
                    while len(inflight) >= MAX_CROP_JOBS:
                        reap_oldest()
                    frames = [
                        {
                            "k": k,
                            "path": str(frame_path(args.frames, seg_id, k)),
                            "people": people[k],
                        }
                        for k in range(N_FRAMES)
                    ]
                    inflight.append(
                        (
                            seg_id,
                            pool.apply_async(
                                crop_segment,
                                (
                                    {
                                        "seg_id": seg_id,
                                        "frames": frames,
                                        "tangent_root": str(args.tangent),
                                        "naive_root": str(args.naive),
                                        "meta_file": str(meta_path(args.meta, seg_id)),
                                    },
                                ),
                            ),
                        )
                    )
                    frames_done += N_FRAMES
                handled = index + 1
                if handled % 20 == 0 or handled == len(pending):
                    elapsed = time.perf_counter() - started
                    rate = frames_done / elapsed if elapsed > 0 else 0.0
                    eta = (len(pending) - handled) * (elapsed / handled) if handled else 0.0
                    print(
                        f"  {handled}/{len(pending)} segments  {rate:.1f} frames/s  "
                        f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}",
                        flush=True,
                    )
    finally:
        while inflight:
            reap_oldest()
        if executor is not None:
            executor.shutdown(wait=True)
        if pool is not None:
            pool.close()
            pool.join()

    print(
        f"Crop jobs: {counts['ok']} ok, {counts['failed']} failed, {counts['crops']} person crops",
        flush=True,
    )

    disk_ids = meta_ids_on_disk(args.meta)
    candidates = []
    for seg_id in disk_ids:
        document = json.loads(meta_path(args.meta, seg_id).read_text(encoding="utf-8"))
        for k, person in iter_people(document):
            candidates.append({"seg_id": seg_id, "k": k, "j": int(person["j"]), "person": person})
    rng = random.Random(args.seed)
    chosen = candidates if len(candidates) <= args.debug_count else rng.sample(candidates, args.debug_count)
    for item in chosen:
        write_debug_image(args.frames, args.tangent, args.naive, args.debug, item)
    print(f"Debug images: {len(chosen)} written to {args.debug}", flush=True)
    print(flush=True)
    print_stats(args.meta, disk_ids)


if __name__ == "__main__":
    main()
