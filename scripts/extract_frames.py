#!/usr/bin/env python3
"""Extract 16 panoramic ERP frames for each protocol segment.

Reads outputs/protocol/segments_v1.csv, finds the 360° ERP video under
datasets/360x/panoramic/ (not binocular or third-person), and writes
data/frames_2048/{seg_id}/{00..15}.jpg. Safe to re-run: a segment is skipped
when its folder already contains 16 JPEGs. Does not modify the protocol CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "outputs" / "protocol" / "segments_v1.csv"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "360x"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "data" / "frames_2048"
DEFAULT_FAILURES = REPO_ROOT / "outputs" / "protocol" / "extract_failures.txt"

N_FRAMES = 16
FRAME_WIDTH = 2048
FRAME_HEIGHT = 1024
JPEG_QUALITY = 90
N_WORKERS = 16
MISSING_VIDEO_LIMIT = 0.05
EXCLUDED_DIR_NAMES = {
    "binocular",
    "monocular",
    "third_person",
    "third-person",
    "front_view",
    "front-view",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    return parser.parse_args()


def load_segments(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No segments in {path}")
    segments = []
    for row in rows:
        segments.append(
            {
                "seg_id": row["seg_id"],
                "video_id": row["video_id"],
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
            }
        )
    return segments


def locate_panoramic_videos(data_root: Path) -> dict[str, Path]:
    """Map video UUID -> ERP mp4. Only paths that contain a panoramic directory."""
    found: dict[str, Path] = {}
    if not data_root.is_dir():
        return found
    for path in data_root.rglob("*.mp4"):
        parts = path.parts
        if "panoramic" not in parts:
            continue
        if EXCLUDED_DIR_NAMES.intersection(parts):
            continue
        video_id = path.stem
        previous = found.get(video_id)
        if previous is None or _video_rank(path) < _video_rank(previous):
            found[video_id] = path
    return found


def _video_rank(path: Path) -> tuple:
    """Prefer datasets/360x/panoramic/{uuid}.mp4 over lower-resolution copies."""
    parts = path.parts
    return (1 if "LR" in parts else 0, len(parts), str(path))


def bin_midpoints(start_s: float, end_s: float, n_frames: int) -> list[float]:
    span = end_s - start_s
    if span <= 0:
        raise ValueError(f"non-positive segment span [{start_s}, {end_s}]")
    return [start_s + (index + 0.5) * span / n_frames for index in range(n_frames)]


def segment_is_complete(folder: Path, n_frames: int) -> bool:
    if not folder.is_dir():
        return False
    return all((folder / f"{index:02d}.jpg").is_file() for index in range(n_frames))


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


def format_elapsed(seconds: float) -> str:
    whole = int(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{int((seconds - whole) * 1000):03d} ({seconds:.1f} s)"


def _failure_line(seg_id: str, video_id: str, reason: str) -> str:
    text = " ".join(str(reason).split())
    return f"{seg_id}\t{video_id}\t{text}\n"


def process_video(task: dict) -> dict:
    """Extract every pending segment of one video. Runs in a worker process."""
    video_id = task["video_id"]
    video_path = task["video_path"]
    frames_root = Path(task["frames_root"])
    n_frames = task["n_frames"]
    size = (task["width"], task["height"])
    quality = task["jpeg_quality"]
    state: dict = {"path": video_path}
    extracted = 0
    skipped = 0
    failures = []
    try:
        for segment in task["segments"]:
            seg_id = segment["seg_id"]
            out_dir = frames_root / str(seg_id)
            if segment_is_complete(out_dir, n_frames):
                skipped += 1
                continue
            try:
                times = bin_midpoints(segment["start_s"], segment["end_s"], n_frames)
                frames = read_segment_frames(state, times)
                write_jpegs(frames, out_dir, size, quality)
                extracted += 1
            except Exception as exc:
                _remove_partial(out_dir, n_frames)
                failures.append(_failure_line(seg_id, video_id, exc).rstrip("\n"))
    finally:
        _close_state(state)
    return {
        "video_id": video_id,
        "extracted": extracted,
        "skipped": skipped,
        "failures": failures,
    }


def read_segment_frames(state: dict, times: list[float]):
    errors = []
    if not state.get("decord_unavailable"):
        try:
            return _read_decord(state, times)
        except ImportError:
            state["decord_unavailable"] = True
        except Exception as exc:
            errors.append(f"decord: {exc}")
            _close_decord(state)
    if not state.get("pyav_unavailable"):
        try:
            return _read_pyav(state, times)
        except ImportError:
            state["pyav_unavailable"] = True
        except Exception as exc:
            errors.append(f"pyav: {exc}")
            _close_pyav(state)
    try:
        return _read_ffmpeg(state, times)
    except Exception as exc:
        errors.append(f"ffmpeg: {exc}")
        raise RuntimeError(" | ".join(errors))


def _read_decord(state: dict, times: list[float]):
    import decord

    decord.bridge.set_bridge("native")
    reader = state.get("decord")
    if reader is None:
        reader = decord.VideoReader(state["path"], num_threads=1)
        state["decord"] = reader
    fps = float(reader.get_avg_fps())
    n_frames = len(reader)
    if not math.isfinite(fps) or fps <= 0 or n_frames <= 0:
        raise RuntimeError(f"invalid decord metadata fps={fps} frames={n_frames}")
    indices = []
    for timestamp in times:
        index = int(round(timestamp * fps))
        indices.append(min(max(index, 0), n_frames - 1))
    batch = reader.get_batch(indices).asnumpy()
    if len(batch) != len(times):
        raise RuntimeError(f"decord returned {len(batch)} frames for {len(times)} timestamps")
    return [frame for frame in batch]


def _read_pyav(state: dict, times: list[float]):
    import av

    container = state.get("pyav")
    if container is None:
        container = av.open(state["path"])
        state["pyav"] = container
        state["pyav_stream"] = container.streams.video[0]
        state["pyav_stream"].thread_type = "AUTO"
        state["pyav_stream"].thread_count = 1
    stream = state["pyav_stream"]
    if stream.time_base is None:
        raise RuntimeError("PyAV stream has no time_base")
    frames = []
    for timestamp in times:
        frames.append(_seek_pyav_frame(container, stream, timestamp))
    return frames


def _seek_pyav_frame(container, stream, timestamp: float):
    seek_to = int(max(timestamp, 0.0) / stream.time_base)
    container.seek(seek_to, any_frame=False, backward=True, stream=stream)
    previous = None
    previous_ts = None
    for frame in container.decode(stream):
        frame_ts = frame.time
        if frame_ts is None and frame.pts is not None:
            frame_ts = float(frame.pts * stream.time_base)
        if frame_ts is None:
            continue
        if frame_ts >= timestamp:
            if previous is not None and abs(previous_ts - timestamp) <= abs(frame_ts - timestamp):
                return previous.to_ndarray(format="rgb24")
            return frame.to_ndarray(format="rgb24")
        previous = frame
        previous_ts = frame_ts
    if previous is not None:
        return previous.to_ndarray(format="rgb24")
    raise RuntimeError(f"PyAV found no frame near t={timestamp:.3f}")


def _read_ffmpeg(state: dict, times: list[float]):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    meta = state.get("ffprobe")
    if meta is None:
        meta = _probe_video(state["path"])
        state["ffprobe"] = meta
    duration = meta.get("duration")
    frames = []
    for timestamp in times:
        clamped = timestamp
        if duration is not None and duration > 0:
            clamped = min(max(timestamp, 0.0), max(duration - 1e-3, 0.0))
        frames.append(_ffmpeg_seek_frame(ffmpeg, state["path"], clamped, meta["width"], meta["height"]))
    return frames


def _probe_video(path: str) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is not on PATH")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"ffprobe failed: {detail}")
    payload = json.loads(completed.stdout.decode("utf-8"))
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe returned no video stream")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    duration_s = float(duration) if duration not in (None, "N/A", "") else None
    return {"width": width, "height": height, "duration": duration_s}


def _ffmpeg_seek_frame(ffmpeg: str, path: str, timestamp: float, width: int, height: int):
    import numpy as np

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    expected = width * height * 3
    if completed.returncode != 0 or len(completed.stdout) < expected:
        detail = completed.stderr.decode("utf-8", errors="replace")[:400]
        raise RuntimeError(
            f"ffmpeg -ss {timestamp:.3f} failed (bytes={len(completed.stdout)} expected={expected}): {detail}"
        )
    frame = np.frombuffer(completed.stdout[:expected], dtype=np.uint8).reshape(height, width, 3)
    return frame.copy()


def write_jpegs(frames, out_dir: Path, size: tuple[int, int], quality: int) -> None:
    from PIL import Image

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for index, frame in enumerate(frames):
            array = frame
            if hasattr(array, "dtype") and str(array.dtype) != "uint8":
                array = array.astype("uint8")
            image = Image.fromarray(array)
            if image.size != size:
                image = image.resize(size, resample)
            destination = out_dir / f"{index:02d}.jpg"
            temporary = out_dir / f"{index:02d}.jpg.part"
            image.save(temporary, format="JPEG", quality=quality)
            os.replace(temporary, destination)
            written.append(destination)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise


def _remove_partial(out_dir: Path, n_frames: int) -> None:
    if not out_dir.exists():
        return
    for index in range(n_frames):
        for path in (out_dir / f"{index:02d}.jpg", out_dir / f"{index:02d}.jpg.part"):
            path.unlink(missing_ok=True)
    try:
        out_dir.rmdir()
    except OSError:
        pass


def _close_decord(state: dict) -> None:
    state.pop("decord", None)


def _close_pyav(state: dict) -> None:
    container = state.pop("pyav", None)
    state.pop("pyav_stream", None)
    if container is not None:
        container.close()


def _close_state(state: dict) -> None:
    _close_decord(state)
    _close_pyav(state)


def report_videos(video_ids: list[str], located: dict[str, Path]) -> list[str]:
    missing = [video_id for video_id in video_ids if video_id not in located]
    found = len(video_ids) - len(missing)
    ratio = (len(missing) / len(video_ids)) if video_ids else 1.0
    print(f"Panoramic ERP videos: {found} found, {len(missing)} missing ({ratio:.1%} of {len(video_ids)})")
    if missing:
        print("Missing:")
        for video_id in missing:
            print(f"  {video_id}")
    else:
        print("Missing: none")
    if ratio > MISSING_VIDEO_LIMIT:
        print(f"More than {MISSING_VIDEO_LIMIT:.0%} of videos are missing. Exiting.")
        sys.exit(1)
    return missing


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    segments = load_segments(args.csv)
    video_ids = sorted({segment["video_id"] for segment in segments})
    located = locate_panoramic_videos(args.data_root)
    missing = set(report_videos(video_ids, located))

    args.frames_root.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)

    pending_by_video = defaultdict(list)
    already_done = 0
    missing_segments = []
    for segment in segments:
        if segment["video_id"] in missing:
            missing_segments.append(segment)
            continue
        out_dir = args.frames_root / str(segment["seg_id"])
        if segment_is_complete(out_dir, args.frames):
            already_done += 1
            continue
        pending_by_video[segment["video_id"]].append(segment)

    tasks = [
        {
            "video_id": video_id,
            "video_path": str(located[video_id]),
            "segments": pending_by_video[video_id],
            "frames_root": str(args.frames_root),
            "n_frames": args.frames,
            "width": args.width,
            "height": args.height,
            "jpeg_quality": args.jpeg_quality,
        }
        for video_id in sorted(pending_by_video)
    ]
    print(
        f"Segments: {len(segments)} total, {already_done} complete, "
        f"{sum(len(task['segments']) for task in tasks)} to extract, "
        f"{len(missing_segments)} missing a video"
    )
    print(f"Workers: {args.workers}")

    failure_lines = [
        _failure_line(segment["seg_id"], segment["video_id"], "panoramic ERP video not found").rstrip("\n")
        for segment in missing_segments
    ]
    extracted = 0
    skipped = 0
    if tasks:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=args.workers) as pool:
            completed = 0
            for result in pool.imap_unordered(process_video, tasks, chunksize=1):
                completed += 1
                extracted += result["extracted"]
                skipped += result["skipped"]
                failure_lines.extend(result["failures"])
                if completed % 10 == 0 or completed == len(tasks):
                    print(
                        f"  videos {completed}/{len(tasks)}  "
                        f"extracted {extracted}  failures {len(failure_lines)}"
                    )

    with args.failures.open("w", encoding="utf-8") as handle:
        for line in failure_lines:
            handle.write(line + "\n")

    elapsed = time.perf_counter() - started
    usage = directory_size(args.frames_root)
    print(f"Failures: {len(failure_lines)} (logged to {args.failures})")
    print(f"Extracted this run: {extracted}  skipped as already complete: {already_done + skipped}")
    print(f"Elapsed: {format_elapsed(elapsed)}")
    print(f"Disk usage ({args.frames_root}): {format_bytes(usage)} ({usage} bytes)")


if __name__ == "__main__":
    main()
