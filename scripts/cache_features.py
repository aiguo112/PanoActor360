#!/usr/bin/env python3
"""Cache frozen DINOv2 features for global, cubemap, or person streams.

Valid segments are folders under data/frames_2048/ that contain exactly 16
frames named 00.jpg..15.jpg. An output file that already exists is skipped.
Cubemap face axis is front, right, back, left, up, down. Does not modify
frames, crops, or protocol files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES = REPO_ROOT / "data" / "frames_2048"
DEFAULT_FEATS = REPO_ROOT / "feats"
DEFAULT_META = REPO_ROOT / "data" / "person_meta"
DEFAULT_TANGENT = REPO_ROOT / "data" / "person_tangent"
DEFAULT_NAIVE = REPO_ROOT / "data" / "person_naive"

MODEL_NAME = "facebook/dinov2-base"
HF_MIRROR = "https://hf-mirror.com"
N_FRAMES = 16
PATCH = 14
GLOBAL_W = 896
GLOBAL_H = 448
GRID_H = GLOBAL_H // PATCH  # 32 latitude patches
GRID_W = GLOBAL_W // PATCH  # 64 longitude patches
POOLED_H = GRID_H // 2  # 16
POOLED_W = GRID_W // 2  # 32
HIDDEN = 768
FACE_W = 224
# py360convert list/dict order: F R B L U D.
FACE_NAMES = ("front", "right", "back", "left", "up", "down")
BATCH_IMAGES = 64
N_WORKERS = 12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, choices=("global", "cubemap", "person"))
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--feats", type=Path, default=DEFAULT_FEATS)
    parser.add_argument("--person-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--tangent", type=Path, default=DEFAULT_TANGENT)
    parser.add_argument("--naive", type=Path, default=DEFAULT_NAIVE)
    parser.add_argument("--batch-size", type=int, default=BATCH_IMAGES)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--suffix", default="", help="Person stream: read data/person_*{suffix}, write feats/person{suffix}")
    return parser.parse_args()


def apply_person_suffix(args: argparse.Namespace) -> str:
    """Return the person feature folder, remapping default crop/meta roots."""
    if args.person_meta == DEFAULT_META:
        args.person_meta = REPO_ROOT / "data" / f"person_meta{args.suffix}"
    if args.tangent == DEFAULT_TANGENT:
        args.tangent = REPO_ROOT / "data" / f"person_tangent{args.suffix}"
    if args.naive == DEFAULT_NAIVE:
        args.naive = REPO_ROOT / "data" / f"person_naive{args.suffix}"
    return f"person{args.suffix}"


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def valid_segment_ids(frames_root: Path) -> tuple[list[str], int]:
    """Folders with exactly the 16 ERP frames 00.jpg..15.jpg."""
    if not frames_root.is_dir():
        raise FileNotFoundError(f"Missing frames directory: {frames_root}")
    expected = {f"{index:02d}.jpg" for index in range(N_FRAMES)}
    kept = []
    skipped = 0
    for folder in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        images = [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        names = {path.name for path in images}
        if len(images) == N_FRAMES and names == expected:
            kept.append(folder.name)
        else:
            skipped += 1
    return kept, skipped


def frame_path(frames_root: Path, seg_id: str, frame_idx: int) -> Path:
    return frames_root / seg_id / f"{frame_idx:02d}.jpg"


def output_path(feats_root: Path, stream: str, seg_id: str) -> Path:
    return feats_root / stream / f"{seg_id}.pt"


def save_pt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _worker_init(_worker_id=None) -> None:
    cv2.setNumThreads(1)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class GlobalDataset(Dataset):
    def __init__(self, frames_root: Path, seg_ids: list[str]):
        self.items = [
            (seg_id, frame_idx, frame_path(frames_root, seg_id, frame_idx))
            for seg_id in seg_ids
            for frame_idx in range(N_FRAMES)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        seg_id, frame_idx, path = self.items[index]
        rgb = read_rgb(path)
        resized = cv2.resize(rgb, (GLOBAL_W, GLOBAL_H), interpolation=cv2.INTER_AREA)
        return normalize_rgb(resized), seg_id, frame_idx


class CubemapDataset(Dataset):
    def __init__(self, frames_root: Path, seg_ids: list[str]):
        self.items = [
            (seg_id, frame_idx, frame_path(frames_root, seg_id, frame_idx))
            for seg_id in seg_ids
            for frame_idx in range(N_FRAMES)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import py360convert

        seg_id, frame_idx, path = self.items[index]
        rgb = read_rgb(path)
        faces = py360convert.e2c(rgb, face_w=FACE_W, mode="bilinear", cube_format="list")
        stacked = torch.stack([normalize_rgb(face) for face in faces], dim=0)
        return stacked, seg_id, frame_idx


class PersonDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        rgb = read_rgb(Path(record["path"]))
        if rgb.shape[0] != FACE_W or rgb.shape[1] != FACE_W:
            rgb = cv2.resize(rgb, (FACE_W, FACE_W), interpolation=cv2.INTER_AREA)
        return normalize_rgb(rgb), record["seg_id"], int(record["slot"]), record["kind"]


def collate_global(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    seg_ids = [item[1] for item in batch]
    frame_idxs = [int(item[2]) for item in batch]
    return images, seg_ids, frame_idxs


def collate_cubemap(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    seg_ids = [item[1] for item in batch]
    frame_idxs = [int(item[2]) for item in batch]
    return images, seg_ids, frame_idxs


def collate_person(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    seg_ids = [item[1] for item in batch]
    slots = [int(item[2]) for item in batch]
    kinds = [item[3] for item in batch]
    return images, seg_ids, slots, kinds


def load_backbone():
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(MODEL_NAME)
    except Exception as first_error:
        print(f"Download of {MODEL_NAME} failed ({first_error}). Retrying with HF_ENDPOINT={HF_MIRROR}", flush=True)
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.requires_grad_(False)
    return model


def forward_hidden(model, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            hidden = model(pixel_values=images).last_hidden_state
    return hidden


def pool_global_tokens(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop CLS, reshape to [32, 64, 768], average-pool 2x2 -> [16, 32, 768].

    Pooled row r is centered at latitude 90 - (r+0.5)*180/16 degrees.
    Pooled col c is centered at longitude -180 + (c+0.5)*360/32 degrees.
    """
    cls = hidden[:, 0].float()
    patches = hidden[:, 1:].float()
    batch, count, channels = patches.shape
    if count != GRID_H * GRID_W or channels != HIDDEN:
        raise RuntimeError(f"expected patch tokens [{GRID_H * GRID_W}, {HIDDEN}], got [{count}, {channels}]")
    grid = patches.reshape(batch, GRID_H, GRID_W, channels)
    pooled = grid.reshape(batch, POOLED_H, 2, POOLED_W, 2, channels).mean(dim=(2, 4))
    return pooled.half().cpu(), cls.half().cpu()


def iter_image_batches(loader, images_per_item: int, batch_images: int):
    """Yield (images [B,3,H,W], metas) with B <= batch_images.

    Global and person samples are one image. A cubemap sample is six faces in
    front, right, back, left, up, down order; those faces are flattened and
    still forwarded in groups of batch_images.
    """
    pending_images = []
    pending_meta = []

    def packed():
        if not pending_images:
            return None
        images = torch.cat(pending_images, dim=0) if len(pending_images) > 1 else pending_images[0]
        return images

    def store(images: torch.Tensor) -> None:
        pending_images.clear()
        if images.shape[0]:
            pending_images.append(images)

    def flush(force: bool):
        nonlocal pending_meta
        images = packed()
        if images is None:
            return
        while images.shape[0] >= batch_images or (force and images.shape[0] > 0):
            take = min(batch_images, images.shape[0])
            yield images[:take], pending_meta[:take]
            images = images[take:]
            pending_meta = pending_meta[take:]
            if images.shape[0] == 0:
                break
        store(images)

    for batch in loader:
        images = batch[0]
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.shape[1] != images_per_item:
            raise RuntimeError(f"expected {images_per_item} images per sample, got {images.shape[1]}")
        batch_size = images.shape[0]
        flat = images.reshape(batch_size * images_per_item, *images.shape[2:])
        if images_per_item == 1 and len(batch) == 3:
            metas = [(seg_id, int(frame_idx), None) for seg_id, frame_idx in zip(batch[1], batch[2])]
        elif images_per_item == 1:
            metas = [(seg_id, int(slot), kind) for seg_id, slot, kind in zip(batch[1], batch[2], batch[3])]
        else:
            metas = []
            for seg_id, frame_idx in zip(batch[1], batch[2]):
                for face_idx in range(images_per_item):
                    metas.append((seg_id, int(frame_idx), face_idx))
        if len(metas) != flat.shape[0]:
            raise RuntimeError(f"meta count {len(metas)} != image count {flat.shape[0]}")
        pending_images.append(flat)
        pending_meta.extend(metas)
        yield from flush(False)
    yield from flush(True)


def make_loader(dataset, collate_fn, workers: int, batch_size: int) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["worker_init_fn"] = _worker_init
    return DataLoader(dataset, **kwargs)


def print_throughput(images_done: int, started: float, files_written: int) -> None:
    elapsed = max(time.perf_counter() - started, 1e-6)
    print(
        f"  throughput: {images_done / elapsed:.1f} images/s  images={images_done}  files_written={files_written}",
        flush=True,
    )


def run_global(loader, model, device, out_dir: Path, batch_images: int) -> tuple[int, int]:
    buffers = {}
    written = 0
    images_done = 0
    started = time.perf_counter()
    step = 0
    for images, metas in iter_image_batches(loader, 1, batch_images):
        pooled, cls = pool_global_tokens(forward_hidden(model, images, device))
        for index, (seg_id, frame_idx, _kind) in enumerate(metas):
            state = buffers.get(seg_id)
            if state is None:
                state = {
                    "tokens": torch.empty(N_FRAMES, POOLED_H, POOLED_W, HIDDEN, dtype=torch.float16),
                    "cls": torch.empty(N_FRAMES, HIDDEN, dtype=torch.float16),
                    "seen": [False] * N_FRAMES,
                }
                buffers[seg_id] = state
            if state["seen"][frame_idx]:
                continue
            state["tokens"][frame_idx] = pooled[index]
            state["cls"][frame_idx] = cls[index]
            state["seen"][frame_idx] = True
            if all(state["seen"]):
                save_pt(out_dir / f"{seg_id}.pt", {"tokens": state["tokens"], "cls": state["cls"]})
                del buffers[seg_id]
                written += 1
        images_done += images.shape[0]
        step += 1
        if step % 10 == 0:
            print_throughput(images_done, started, written)
    if buffers:
        raise RuntimeError(f"{len(buffers)} global segments were missing frames")
    print_throughput(images_done, started, written)
    return written, images_done


def run_cubemap(loader, model, device, out_dir: Path, batch_images: int) -> tuple[int, int]:
    buffers = {}
    written = 0
    images_done = 0
    started = time.perf_counter()
    step = 0
    for images, metas in iter_image_batches(loader, len(FACE_NAMES), batch_images):
        cls = forward_hidden(model, images, device)[:, 0].half().cpu()
        for index, (seg_id, frame_idx, face_idx) in enumerate(metas):
            state = buffers.get(seg_id)
            if state is None:
                state = {
                    "cls": torch.empty(N_FRAMES, len(FACE_NAMES), HIDDEN, dtype=torch.float16),
                    "seen": [[False] * len(FACE_NAMES) for _ in range(N_FRAMES)],
                    "count": 0,
                }
                buffers[seg_id] = state
            if state["seen"][frame_idx][face_idx]:
                continue
            state["cls"][frame_idx, face_idx] = cls[index]
            state["seen"][frame_idx][face_idx] = True
            state["count"] += 1
            if state["count"] == N_FRAMES * len(FACE_NAMES):
                save_pt(out_dir / f"{seg_id}.pt", {"cls": state["cls"]})
                del buffers[seg_id]
                written += 1
        images_done += images.shape[0]
        step += 1
        if step % 10 == 0:
            print_throughput(images_done, started, written)
    if buffers:
        raise RuntimeError(f"{len(buffers)} cubemap segments were missing faces")
    print_throughput(images_done, started, written)
    return written, images_done


def load_person_records(seg_id: str, meta_root: Path, tangent_root: Path, naive_root: Path):
    path = meta_root / f"{seg_id}.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    frames = document.get("frames") or []
    people = []
    for frame_idx, frame_people in enumerate(frames):
        for person in frame_people:
            j = int(person["j"])
            tangent = tangent_root / seg_id / f"{frame_idx:02d}_{j}.jpg"
            naive = naive_root / seg_id / f"{frame_idx:02d}_{j}.jpg"
            people.append(
                {
                    "frame_idx": frame_idx,
                    "lat": float(person["lat"]),
                    "lon": float(person["lon"]),
                    "fov": float(person["fov"]),
                    "conf": float(person["conf"]),
                    "seam": int(person["seam"]),
                    "tangent": str(tangent),
                    "naive": str(naive),
                }
            )
    return people


def empty_person_payload() -> dict:
    return {
        "tangent": torch.empty((0, HIDDEN), dtype=torch.float16),
        "naive": torch.empty((0, HIDDEN), dtype=torch.float16),
        "frame_idx": torch.empty((0,), dtype=torch.int64),
        "lat": torch.empty((0,), dtype=torch.float32),
        "lon": torch.empty((0,), dtype=torch.float32),
        "fov": torch.empty((0,), dtype=torch.float32),
        "conf": torch.empty((0,), dtype=torch.float32),
        "seam": torch.empty((0,), dtype=torch.int64),
    }


def person_meta_tensors(people: list[dict]) -> dict:
    return {
        "frame_idx": torch.tensor([person["frame_idx"] for person in people], dtype=torch.int64),
        "lat": torch.tensor([person["lat"] for person in people], dtype=torch.float32),
        "lon": torch.tensor([person["lon"] for person in people], dtype=torch.float32),
        "fov": torch.tensor([person["fov"] for person in people], dtype=torch.float32),
        "conf": torch.tensor([person["conf"] for person in people], dtype=torch.float32),
        "seam": torch.tensor([person["seam"] for person in people], dtype=torch.int64),
    }


def run_person(loader, model, device, out_dir: Path, batch_images: int, states: dict) -> tuple[int, int]:
    written = 0
    images_done = 0
    started = time.perf_counter()
    step = 0
    for images, metas in iter_image_batches(loader, 1, batch_images):
        cls = forward_hidden(model, images, device)[:, 0].half().cpu()
        for index, (seg_id, slot, kind) in enumerate(metas):
            state = states[seg_id]
            bucket = state[kind]
            if state["filled"][kind][slot]:
                continue
            bucket[slot] = cls[index]
            state["filled"][kind][slot] = True
            state["count"] += 1
            if state["count"] == state["target"]:
                save_pt(
                    out_dir / f"{seg_id}.pt",
                    {
                        "tangent": state["tangent"],
                        "naive": state["naive"],
                        **state["meta"],
                    },
                )
                del states[seg_id]
                written += 1
        images_done += images.shape[0]
        step += 1
        if step % 10 == 0:
            print_throughput(images_done, started, written)
    if states:
        missing = ", ".join(sorted(states)[:8])
        raise RuntimeError(f"{len(states)} person segments were incomplete, including {missing}")
    if images_done:
        print_throughput(images_done, started, written)
    return written, images_done


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for fp16 autocast")
    device = torch.device("cuda")

    seg_ids, invalid = valid_segment_ids(args.frames)
    stream_dir = apply_person_suffix(args) if args.stream == "person" else args.stream
    out_dir = args.feats / stream_dir
    pending = [seg_id for seg_id in seg_ids if not output_path(args.feats, stream_dir, seg_id).is_file()]
    print(
        f"Stream: {stream_dir}  valid segments: {len(seg_ids)}  "
        f"skipped invalid folders: {invalid}  already cached: {len(seg_ids) - len(pending)}  pending: {len(pending)}",
        flush=True,
    )

    written = 0
    images_done = 0
    model = None
    loader = None

    if args.stream == "global":
        if pending:
            dataset = GlobalDataset(args.frames, pending)
            loader = make_loader(dataset, collate_global, args.workers, args.batch_size)
    elif args.stream == "cubemap":
        if pending:
            dataset = CubemapDataset(args.frames, pending)
            loader_batch = max(1, args.batch_size // len(FACE_NAMES))
            loader = make_loader(dataset, collate_cubemap, args.workers, loader_batch)
    elif args.stream == "person":
        records = []
        states = {}
        missing_meta = []
        missing_crops = []
        for seg_id in pending:
            people = load_person_records(seg_id, args.person_meta, args.tangent, args.naive)
            if people is None:
                missing_meta.append(seg_id)
                continue
            if any(not Path(person["tangent"]).is_file() or not Path(person["naive"]).is_file() for person in people):
                missing_crops.append(seg_id)
                continue
            if not people:
                save_pt(out_dir / f"{seg_id}.pt", empty_person_payload())
                written += 1
                continue
            count = len(people)
            states[seg_id] = {
                "tangent": torch.empty(count, HIDDEN, dtype=torch.float16),
                "naive": torch.empty(count, HIDDEN, dtype=torch.float16),
                "filled": {
                    "tangent": [False] * count,
                    "naive": [False] * count,
                },
                "count": 0,
                "target": count * 2,
                "meta": person_meta_tensors(people),
            }
            for slot, person in enumerate(people):
                for kind in ("tangent", "naive"):
                    records.append({"seg_id": seg_id, "slot": slot, "kind": kind, "path": person[kind]})
        if missing_meta:
            print(f"Person segments missing meta JSON (not written): {len(missing_meta)}", flush=True)
        if missing_crops:
            print(f"Person segments missing crops (not written): {len(missing_crops)}", flush=True)
        if records:
            dataset = PersonDataset(records)
            loader = make_loader(dataset, collate_person, args.workers, args.batch_size)
            person_states = states
        else:
            person_states = {}
    else:
        raise ValueError(args.stream)

    if loader is not None:
        print(f"Backbone: frozen {MODEL_NAME}  batch={args.batch_size} images  workers={args.workers}", flush=True)
        model = load_backbone().to(device)
        if args.stream == "global":
            n_written, images_done = run_global(loader, model, device, out_dir, args.batch_size)
        elif args.stream == "cubemap":
            n_written, images_done = run_cubemap(loader, model, device, out_dir, args.batch_size)
        else:
            n_written, images_done = run_person(loader, model, device, out_dir, args.batch_size, person_states)
        written += n_written

    usage = directory_size(out_dir)
    print(f"Files written: {written}", flush=True)
    print(f"Images forwarded: {images_done}", flush=True)
    print(f"Disk usage ({out_dir}): {format_bytes(usage)} ({usage} bytes)", flush=True)


if __name__ == "__main__":
    main()
