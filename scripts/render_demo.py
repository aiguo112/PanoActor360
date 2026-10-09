#!/usr/bin/env python3
"""Render a supplementary demo video of PanoActor detections and actor attention.

Left panel: ERP frames with wrap-aware boxes colored by MIL attention alpha
(same YlOrRd map as Fig. 3). Right panel: tangent crop of the highest-alpha
person on that frame, with GT label and top-3 predictions. Faces are blurred.
Writes paper_fg2027/supp/panoactor_demo.mp4 via ffmpeg.

Uses fold-0 full-model predictions and alpha dumps. Does not redistribute raw
360+x; output is for reviewer supplement only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_PATH = REPO_ROOT / "results" / "full" / "alpha_fold0.json"
PREDS_PATH = REPO_ROOT / "results" / "full" / "preds_fold0.json"
CLASSES_PATH = REPO_ROOT / "outputs" / "protocol" / "classes_v1.json"
META_ROOT = REPO_ROOT / "data" / "person_meta"
FRAMES_ROOT = REPO_ROOT / "data" / "frames_2048"
TANGENT_ROOT = REPO_ROOT / "data" / "person_tangent"
OUT_DIR = REPO_ROOT / "paper_fg2027" / "supp"
OUT_MP4 = OUT_DIR / "panoactor_demo.mp4"

# Reuse wrap / crop helpers from the qualitative figure script.
import sys

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_fig_qual import (  # noqa: E402
    load_rgb,
    match_person_j,
    wrap_rects,
)


def blur_region(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    height, width = rgb.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    region = rgb[y1:y2, x1:x2]
    k = max(15, (min(region.shape[0], region.shape[1]) // 2) | 1)
    rgb[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), k / 3.0)


def blur_faces_safe(rgb: np.ndarray, person_boxes: list[list[float]] | None = None) -> np.ndarray:
    """Blur faces; fall back to the top 25% of each person box if detectors fail."""
    out = rgb.copy()
    height, width = out.shape[:2]
    try:
        from make_fig_qual import detect_faces

        faces = detect_faces(rgb, person_boxes)
        n_blurred = 0
        for x, y, w, h in faces:
            if not np.isfinite([x, y, w, h]).all() or w < 4 or h < 4:
                continue
            pad_x = int(round(0.18 * w))
            pad_y = int(round(0.22 * h))
            blur_region(out, int(x) - pad_x, int(y) - pad_y, int(x + w) + pad_x, int(y + h) + pad_y)
            n_blurred += 1
        if n_blurred:
            return out
    except Exception as error:
        print(f"face detector fallback ({error})", flush=True)
    for box in person_boxes or []:
        for x1, y1, x2, y2 in wrap_rects(box, width):
            bw = x2 - x1
            bh = y2 - y1
            if bw < 8 or bh < 8:
                continue
            blur_region(out, int(x1), int(y1), int(x2), int(y1 + 0.25 * bh))
    if not person_boxes:
        blur_region(out, width // 4, 0, 3 * width // 4, height // 3)
    return out

ERP_W = 2048
ERP_H = 1024
PANEL_W = 1280
PANEL_H = 640
FPS = 4
HOLD_TITLE = 10  # frames of title card
CMAP_NAME = "YlOrRd"
PAPER_EXAMPLES = ("88", "3372", "2346", "3186")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_MP4)
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def class_name(classes: dict, index: int) -> str:
    return classes[str(int(index))]


def geometry(seg_id: str) -> dict:
    meta = json.loads((META_ROOT / f"{seg_id}.json").read_text(encoding="utf-8"))
    fovs = []
    has_seam = False
    max_people = 0
    for frame in meta.get("frames") or []:
        max_people = max(max_people, len(frame))
        for person in frame:
            fovs.append(float(person["fov"]))
            if int(person.get("seam", 0)) == 1:
                has_seam = True
    return {
        "has_seam": has_seam,
        "max_fov": max(fovs) if fovs else 0.0,
        "max_people": max_people,
        "n_dets": len(fovs),
    }


def alpha_mass(segment: dict) -> float:
    if not segment["persons"]:
        return 0.0
    return float(max(float(p["alpha"]) for p in segment["persons"]))


def select_clips(alpha: dict, preds: dict[str, dict], limit: int) -> list[dict]:
    segments = {str(s["seg_id"]): s for s in alpha["segments"]}
    geo = {seg_id: geometry(seg_id) for seg_id in segments}
    chosen: list[dict] = []
    used: set[str] = set()

    def add(seg_id: str, tag: str, note: str) -> None:
        if seg_id in used or seg_id not in segments or len(chosen) >= limit:
            return
        segment = segments[seg_id]
        pred = preds.get(seg_id)
        if pred is None:
            return
        chosen.append(
            {
                "seg_id": seg_id,
                "tag": tag,
                "note": note,
                "segment": segment,
                "pred": pred,
                "geo": geo[seg_id],
            }
        )
        used.add(seg_id)

    for seg_id in PAPER_EXAMPLES:
        g = geo.get(seg_id)
        if g is None:
            continue
        tag = "seam" if g["has_seam"] else "success"
        add(seg_id, tag, "paper figure example")

    correct = [
        sid
        for sid, seg in segments.items()
        if int(seg["predicted"]) == int(seg["true"]) and sid not in used
    ]
    correct.sort(key=lambda sid: (-geo[sid]["has_seam"], -geo[sid]["max_fov"], -alpha_mass(segments[sid])))
    for sid in correct:
        if geo[sid]["has_seam"] and geo[sid]["max_fov"] >= 60:
            add(sid, "seam+near", "seam-crossing near-field actor")
        if len(chosen) >= limit - 4:
            break
    for sid in correct:
        if geo[sid]["max_people"] >= 8 and geo[sid]["max_fov"] >= 60:
            add(sid, "crowded", "crowded near-field scene")
        if len(chosen) >= limit - 3:
            break
    for sid in correct:
        if geo[sid]["max_fov"] >= 70:
            add(sid, "near", "near-field person")
        if len(chosen) >= limit - 2:
            break

    failures = [
        sid
        for sid, seg in segments.items()
        if int(seg["predicted"]) != int(seg["true"]) and sid not in used and geo[sid]["n_dets"] > 0
    ]
    failures.sort(key=lambda sid: -alpha_mass(segments[sid]))
    for sid in failures[:2]:
        add(sid, "failure", "incorrect prediction")

    for sid in correct:
        add(sid, "success", "correct prediction")
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def resize_panel(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def color_alpha(alpha: float, cmap) -> tuple[int, int, int]:
    r, g, b, _a = cmap(float(np.clip(alpha, 0.0, 1.0)))
    return int(255 * r), int(255 * g), int(255 * b)


def draw_erp_panel(seg_id: str, frame_idx: int, people: list[dict], cmap) -> np.ndarray:
    frame_path = FRAMES_ROOT / str(seg_id) / f"{frame_idx:02d}.jpg"
    rgb = load_rgb(frame_path)
    boxes = [person["box"] for person in people if person.get("box")]
    rgb = blur_faces_safe(rgb, boxes)
    canvas = rgb.copy()
    ranked = sorted(people, key=lambda p: float(p["alpha"]))
    top_alpha = max((float(p["alpha"]) for p in people), default=0.0)
    for person in ranked:
        if not person.get("box"):
            continue
        color = color_alpha(float(person["alpha"]), cmap)
        is_top = abs(float(person["alpha"]) - top_alpha) < 1e-12 and top_alpha > 0
        thickness = 5 if is_top else 2
        for x1, y1, x2, y2 in wrap_rects(person["box"], ERP_W):
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(canvas, p1, p2, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.rectangle(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
            if is_top:
                label = f"a={float(person['alpha']):.2f}"
                cv2.putText(
                    canvas,
                    label,
                    (p1[0] + 4, max(p1[1] - 8, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    label,
                    (p1[0] + 4, max(p1[1] - 8, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    # Letterbox ERP 2:1 into 1280x640.
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    scaled = resize_panel(canvas, PANEL_W, PANEL_H)
    panel[:] = scaled
    cv2.putText(
        panel,
        f"ERP  frame {frame_idx:02d}/15",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def load_tangent(seg_id: str, person: dict) -> np.ndarray | None:
    if not person.get("box"):
        return None
    frame_idx = int(person["frame_idx"])
    person_j = match_person_j(seg_id, frame_idx, person["box"])
    path = TANGENT_ROOT / str(seg_id) / f"{frame_idx:02d}_{person_j}.jpg"
    if not path.is_file():
        return None
    return blur_faces_safe(load_rgb(path))


def clip_top_person(segment: dict) -> dict | None:
    people = [person for person in segment["persons"] if person.get("box")]
    if not people:
        return None
    return max(people, key=lambda person: float(person["alpha"]))


def person_on_frame(people: list[dict], actor: dict | None, lon_tol: float = 0.35) -> dict | None:
    """Prefer the clip-level actor if they appear on this frame (matched by lon/lat)."""
    if actor is None:
        return max(people, key=lambda person: float(person["alpha"])) if people else None
    best = None
    best_d = float("inf")
    for person in people:
        d_lon = abs(float(person["lon"]) - float(actor["lon"]))
        d_lon = min(d_lon, abs(d_lon - 2 * np.pi))
        dist = d_lon + abs(float(person["lat"]) - float(actor["lat"]))
        if dist < best_d:
            best_d = dist
            best = person
    if best is not None and best_d < lon_tol:
        return best
    return max(people, key=lambda person: float(person["alpha"])) if people else None


def draw_right_panel(
    seg_id: str,
    frame_idx: int,
    people: list[dict],
    clip: dict,
    classes: dict,
    actor: dict | None,
) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W, 3), 24, dtype=np.uint8)
    # Always show the clip-level highest-alpha person's crop.
    on_frame = person_on_frame(people, actor)
    crop = load_tangent(seg_id, actor) if actor is not None else None
    crop_h = PANEL_H - 150
    crop_w = min(PANEL_W - 80, int(crop_h * 0.85))
    x0 = (PANEL_W - crop_w) // 2
    y0 = 40
    if crop is not None and actor is not None:
        shown = resize_panel(crop, crop_w, crop_h)
        panel[y0 : y0 + crop_h, x0 : x0 + crop_w] = shown
        cv2.rectangle(panel, (x0, y0), (x0 + crop_w, y0 + crop_h), (220, 220, 220), 2)
        matched = (
            on_frame is not None
            and abs(float(on_frame["lon"]) - float(actor["lon"])) < 0.35
            and abs(float(on_frame["lat"]) - float(actor["lat"])) < 0.35
        )
        status = f"on frame {frame_idx:02d}" if matched else f"peak frame {int(actor['frame_idx']):02d}"
        cv2.putText(
            panel,
            f"actor a={float(actor['alpha']):.2f}  ({status})",
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            panel,
            "no person detections",
            (80, PANEL_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

    pred = clip["pred"]
    gt = class_name(classes, pred["true"])
    top3 = [class_name(classes, index) for index in pred["top5"][:3]]
    hit = int(pred["pred"]) == int(pred["true"])
    status = "CORRECT" if hit else "FAILURE"
    status_color = (80, 200, 120) if hit else (80, 80, 220)
    bar_y = PANEL_H - 130
    cv2.rectangle(panel, (0, bar_y), (PANEL_W, PANEL_H), (12, 12, 12), -1)
    cv2.putText(panel, status, (24, bar_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"GT: {gt}",
        (24, bar_y + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Top-3: {top3[0]}  |  {top3[1]}  |  {top3[2]}",
        (24, bar_y + 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    return panel


def title_card(clip: dict, classes: dict, index: int, total: int) -> np.ndarray:
    frame = np.full((PANEL_H, PANEL_W * 2, 3), 18, dtype=np.uint8)
    pred = clip["pred"]
    gt = class_name(classes, pred["true"])
    lines = [
        "PanoActor  —  supplementary demo",
        f"Clip {index}/{total}   id {clip['seg_id']}   [{clip['tag']}]",
        clip["note"],
        f"Ground truth: {gt}",
        f"Prediction: {class_name(classes, pred['pred'])}",
        f"Seam={clip['geo']['has_seam']}  max FOV={clip['geo']['max_fov']:.0f}  "
        f"max people/frame={clip['geo']['max_people']}",
    ]
    y = 160
    for i, line in enumerate(lines):
        scale = 1.35 if i == 0 else 0.95
        color = (240, 240, 240) if i != 0 else (255, 210, 120)
        cv2.putText(frame, line, (80, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 70 if i == 0 else 55
    return frame


def opening_card() -> np.ndarray:
    frame = np.full((PANEL_H, PANEL_W * 2, 3), 18, dtype=np.uint8)
    lines = [
        "PanoActor",
        "Person-centric action recognition in 360 video",
        "Left: ERP + wrap-aware detections colored by actor attention",
        "Right: tangent crop of the highest-attention person",
        "Faces blurred. Demo only; not a raw dataset redistribution.",
    ]
    y = 180
    for i, line in enumerate(lines):
        scale = 1.8 if i == 0 else 0.95
        color = (255, 210, 120) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (80, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 80 if i == 0 else 55
    return frame


def write_bgr(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise SystemExit(f"failed to write {path}")


def encode_mp4(frame_dir: Path, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MB)", flush=True)


def render_clip_frames(clip: dict, classes: dict, cmap, frame_dir: Path, counter: list[int]) -> None:
    seg_id = clip["seg_id"]
    segment = clip["segment"]
    by_frame: dict[int, list[dict]] = {index: [] for index in range(16)}
    for person in segment["persons"]:
        if person.get("box"):
            by_frame[int(person["frame_idx"])].append(person)

    actor = clip_top_person(segment)
    for frame_idx in range(16):
        people = by_frame[frame_idx]
        left = draw_erp_panel(seg_id, frame_idx, people, cmap)
        right = draw_right_panel(seg_id, frame_idx, people, clip, classes, actor)
        combo = np.concatenate([left, right], axis=1)
        path = frame_dir / f"{counter[0]:06d}.jpg"
        write_bgr(path, combo)
        counter[0] += 1


def main() -> None:
    args = parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    alpha = json.loads(ALPHA_PATH.read_text(encoding="utf-8"))
    preds = {str(row["seg_id"]): row for row in json.loads(PREDS_PATH.read_text(encoding="utf-8"))}
    clips = select_clips(alpha, preds, args.clips)
    if len(clips) < 3:
        raise SystemExit(f"only selected {len(clips)} clips")
    print("selected clips:", flush=True)
    for index, clip in enumerate(clips, start=1):
        pred = clip["pred"]
        print(
            f"  {index:02d} id={clip['seg_id']} tag={clip['tag']} "
            f"gt={class_name(classes, pred['true'])} pred={class_name(classes, pred['pred'])} "
            f"seam={clip['geo']['has_seam']} fov={clip['geo']['max_fov']:.0f} "
            f"people={clip['geo']['max_people']}",
            flush=True,
        )

    cmap = plt.colormaps[CMAP_NAME]
    temporary = Path(tempfile.mkdtemp(prefix="panoactor_demo_"))
    counter = [0]
    try:
        opening = opening_card()
        for _ in range(HOLD_TITLE):
            write_bgr(temporary / f"{counter[0]:06d}.jpg", opening)
            counter[0] += 1
        for index, clip in enumerate(clips, start=1):
            card = title_card(clip, classes, index, len(clips))
            # Patch the title card to use real index/total (already passed).
            for _ in range(HOLD_TITLE):
                write_bgr(temporary / f"{counter[0]:06d}.jpg", card)
                counter[0] += 1
            print(f"rendering clip {index}/{len(clips)} id={clip['seg_id']}", flush=True)
            render_clip_frames(clip, classes, cmap, temporary, counter)
        encode_mp4(temporary, args.out, args.fps)
        manifest = args.out.with_suffix(".json")
        payload = [
            {
                "seg_id": clip["seg_id"],
                "tag": clip["tag"],
                "note": clip["note"],
                "true": int(clip["pred"]["true"]),
                "pred": int(clip["pred"]["pred"]),
                "top5": clip["pred"]["top5"],
                "geo": clip["geo"],
            }
            for clip in clips
        ]
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {manifest}", flush=True)
    finally:
        if args.keep_frames:
            kept = args.out.parent / "demo_frames"
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(temporary, kept)
            print(f"kept frames at {kept}", flush=True)
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
