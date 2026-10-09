#!/usr/bin/env python3
"""Build a 5-fold temporal-action protocol from outputs/action_segments.csv.

Drops incomplete and sub-second segments, keeps classes that still have at
least 5 videos and 20 segments, then assigns folds with StratifiedGroupKFold
so every segment from a video stays in one fold. Writes new files under
outputs/protocol/ and does not modify the input CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "action_segments.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "protocol"

OUT_FIELDS = [
    "seg_id",
    "video_id",
    "start_s",
    "end_s",
    "duration",
    "action_label",
    "class_idx",
    "fold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--min-videos", type=int, default=5)
    parser.add_argument("--min-segments", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row_index, row in enumerate(reader):
            row["seg_id"] = row_index
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def parse_time(value: str):
    text = (value or "").strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(text)


def filter_duration(rows: list[dict], min_duration: float) -> list[dict]:
    kept = []
    for row in rows:
        start_s = parse_time(row.get("start_s", ""))
        end_s = parse_time(row.get("end_s", ""))
        if start_s is None or end_s is None:
            continue
        duration = end_s - start_s
        if duration < min_duration:
            continue
        kept.append(
            {
                "seg_id": row["seg_id"],
                "video_id": row["video_id"],
                "start_s": start_s,
                "end_s": end_s,
                "duration": duration,
                "action_label": row["action_label"],
            }
        )
    return kept


def filter_classes(rows: list[dict], min_videos: int, min_segments: int) -> tuple[list[dict], list[str]]:
    segment_counts = Counter(row["action_label"] for row in rows)
    videos = defaultdict(set)
    for row in rows:
        videos[row["action_label"]].add(row["video_id"])
    labels = sorted(
        label
        for label, count in segment_counts.items()
        if count >= min_segments and len(videos[label]) >= min_videos
    )
    keep = set(labels)
    return [row for row in rows if row["action_label"] in keep], labels


def assign_folds(rows: list[dict], n_splits: int, seed: int) -> None:
    labels = np.array([row["action_label"] for row in rows])
    groups = np.array([row["video_id"] for row in rows])
    features = np.zeros((len(rows), 1))
    folds = np.full(len(rows), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (_, test_index) in enumerate(splitter.split(features, labels, groups)):
        folds[test_index] = fold_id
    if np.any(folds < 0):
        raise RuntimeError("StratifiedGroupKFold left some segments without a fold")
    for row, fold_id in zip(rows, folds.tolist()):
        row["fold"] = int(fold_id)

    folds_by_video = defaultdict(set)
    for row in rows:
        folds_by_video[row["video_id"]].add(row["fold"])
    mixed = [video_id for video_id, seen in folds_by_video.items() if len(seen) != 1]
    if mixed:
        raise RuntimeError(f"{len(mixed)} videos were split across folds")


def write_segments(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["seg_id"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "seg_id": row["seg_id"],
                    "video_id": row["video_id"],
                    "start_s": _format_number(row["start_s"]),
                    "end_s": _format_number(row["end_s"]),
                    "duration": _format_number(row["duration"]),
                    "action_label": row["action_label"],
                    "class_idx": row["class_idx"],
                    "fold": row["fold"],
                }
            )


def write_classes(path: Path, labels: list[str]) -> None:
    mapping = {str(index): label for index, label in enumerate(labels)}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)
        handle.write("\n")


def print_report(rows: list[dict], labels: list[str], n_splits: int) -> None:
    fold_counts = Counter(row["fold"] for row in rows)
    class_fold = Counter((row["class_idx"], row["fold"]) for row in rows)
    width = max(len(label) for label in labels)

    print(f"Final class count: {len(labels)}")
    print(f"Total segments: {len(rows)}")
    print("Segments per fold:")
    for fold_id in range(n_splits):
        print(f"  fold {fold_id}: {fold_counts[fold_id]}")
    print()

    header = f"  {'class':<{width}}  {'idx':>3}  " + "  ".join(f"{fold_id:>6}" for fold_id in range(n_splits)) + f"  {'total':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    empty = []
    for class_idx, label in enumerate(labels):
        counts = [class_fold[(class_idx, fold_id)] for fold_id in range(n_splits)]
        total = sum(counts)
        cells = "  ".join(f"{count:6d}" for count in counts)
        print(f"  {label:<{width}}  {class_idx:3d}  {cells}  {total:6d}")
        missing = [fold_id for fold_id, count in enumerate(counts) if count == 0]
        if missing:
            empty.append((label, missing))

    print()
    print(f"Classes with 0 segments in any fold: {len(empty)}")
    if empty:
        for label, missing in empty:
            folds = ", ".join(str(fold_id) for fold_id in missing)
            print(f"  {label}: fold {folds}")
    else:
        print("  none")


def _format_number(value: float) -> str:
    return f"{value:.5f}".rstrip("0").rstrip(".")


def main() -> None:
    args = parse_args()
    source_rows = load_rows(args.input)
    timed = filter_duration(source_rows, args.min_duration)
    kept, labels = filter_classes(timed, args.min_videos, args.min_segments)
    if not kept:
        raise ValueError("No segments left after class filtering")

    class_to_idx = {label: index for index, label in enumerate(labels)}
    for row in kept:
        row["class_idx"] = class_to_idx[row["action_label"]]

    print(f"Input rows: {len(source_rows)}")
    print(f"After dropping empty end_s and duration < {args.min_duration:.1f}s: {len(timed)}")
    print(
        f"After keeping classes with >= {args.min_videos} videos and "
        f">= {args.min_segments} segments: {len(kept)} segments, {len(labels)} classes"
    )
    print("seg_id is the 0-based row index in the input CSV.")
    print()

    assign_folds(kept, args.n_splits, args.seed)

    segments_path = args.out_dir / "segments_v1.csv"
    classes_path = args.out_dir / "classes_v1.json"
    write_segments(segments_path, kept)
    write_classes(classes_path, labels)
    print(f"Wrote {segments_path}")
    print(f"Wrote {classes_path}")
    print()
    print_report(kept, labels, args.n_splits)


if __name__ == "__main__":
    main()
