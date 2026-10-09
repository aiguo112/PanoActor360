#!/usr/bin/env python3
"""Measure parameter count, GMACs, and latency of the recognition pipeline.

DINOv2 and the trainable head use fvcore FlopCountAnalysis. That counter
counts multiplications (MACs) and skips elementwise ops; scaled-dot-product
attention is added with the same multiply count (two batched matmuls).
Ultralytics model.info FLOPs are multiply-adds, so they are divided by 2.
Every reported compute number is GMACs. Clip parameter counts add each
distinct network once. full / full_light clip cost uses one batched C75
call rather than 75 single-crop forwards.

Latency is the median of 50 forwards after 10 warmup iterations, in fp16
autocast, eval mode, and torch.no_grad, with a CUDA synchronize around each
timed call.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fvcore.nn import FlopCountAnalysis  # noqa: E402
from fvcore.nn.jit_handles import get_shape  # noqa: E402
from har.model import FEAT_DIM, GRID_H, GRID_W, HARModel, N_FRAMES  # noqa: E402

RESULTS_PATH = REPO_ROOT / "results" / "cost.csv"
WARMUP = 10
RUNS = 50
N_PERSONS = 75
HEAD_CONFIGS = ("erp_cls_mean", "cubemap", "grid_latlon", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_configs() -> dict:
    spec = importlib.util.spec_from_file_location("train_har_configs", REPO_ROOT / "scripts" / "train_har.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIGS


def param_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def sdpa_flop_jit(inputs, outputs) -> int:
    """Multiply count of QK^T and AV. Query layout is [batch, heads, seq, dim]."""
    query = get_shape(inputs[0])
    if query is None or len(query) != 4:
        raise RuntimeError(f"unexpected SDPA query shape {query}")
    batch, heads, seq, dim = query
    return int(2 * batch * heads * seq * seq * dim)


def flop_analysis(module: nn.Module, example) -> FlopCountAnalysis:
    analysis = FlopCountAnalysis(module, example)
    analysis.set_op_handle("aten::scaled_dot_product_attention", sdpa_flop_jit)
    return analysis


def gflops(module: nn.Module, example) -> float:
    analysis = flop_analysis(module, example)
    total = analysis.total()
    missed = {
        name: count
        for name, count in analysis.unsupported_ops().items()
        if "attention" in name or name in {"aten::matmul", "aten::mm", "aten::bmm", "aten::linear"}
    }
    if missed:
        raise RuntimeError(f"FLOP count missed attention or matmul ops: {missed}")
    return total / 1e9


def latency_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def autocast_call(fn):
    def wrapped():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fn()

    return wrapped


class TensorCall(nn.Module):
    """Trace a dict-input module from a fixed list of tensor arguments."""

    def __init__(self, module: nn.Module, keys: tuple[str, ...]):
        super().__init__()
        self.module = module
        self.keys = keys

    def forward(self, *tensors):
        return self.module({key: value for key, value in zip(self.keys, tensors)})


class DinoForward(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.module(pixel_values=pixel_values).last_hidden_state


def load_dinov2():
    from transformers import AutoModel

    name = "facebook/dinov2-base"
    try:
        model = AutoModel.from_pretrained(name)
    except Exception as first_error:
        print(f"Download of {name} failed ({first_error}). Retrying with the HF mirror.", flush=True)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model = AutoModel.from_pretrained(name)
    model.eval()
    model.requires_grad_(False)
    return model


def yolo_gmacs(model, image_hw: tuple[int, int]) -> tuple[int, float]:
    """Ultralytics model.info GFLOPs at the rectangular size, converted to GMACs."""
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    height, width = image_hw
    flops = get_flops(model, imgsz=[height, width])
    if not flops:
        raise RuntimeError(f"ultralytics get_flops returned {flops} for image {width}x{height}")
    return get_num_params(model), float(flops) / 2.0


def warn_if_gpu_busy(gpu_index: int) -> None:
    """Warn when nvidia-smi lists another compute process on this GPU."""
    try:
        listed = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
        uuid_map = {}
        for line in listed.splitlines():
            if not line.strip():
                continue
            index_text, uuid = [part.strip() for part in line.split(",", 1)]
            uuid_map[int(index_text)] = uuid
        uuid = uuid_map.get(gpu_index)
        if uuid is None:
            return
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        print(f"WARNING: could not query nvidia-smi ({error})", flush=True)
        return
    others = []
    self_pid = os.getpid()
    for line in apps.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        app_uuid, pid_text, name = parts[0], parts[1], parts[2]
        if app_uuid != uuid:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        memory = parts[3] if len(parts) > 3 else "?"
        others.append(f"{name} pid={pid} {memory} MiB")
    if others:
        print(f"WARNING: other processes are using GPU {gpu_index}: {'; '.join(others)}", flush=True)


def head_example(config: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch = {"frame_mask": torch.ones(1, N_FRAMES, dtype=torch.bool, device=device)}
    stream = config["global_stream"]
    if stream == "cls":
        batch["cls"] = torch.randn(1, N_FRAMES, FEAT_DIM, device=device)
    elif stream == "cubemap":
        batch["faces"] = torch.randn(1, N_FRAMES, 6, FEAT_DIM, device=device)
    elif stream == "grid":
        batch["grid"] = torch.randn(1, N_FRAMES, GRID_H, GRID_W, FEAT_DIM, device=device)
    if config["persons"] is not None:
        frames = torch.arange(N_PERSONS, device=device).remainder(N_FRAMES)
        batch.update(
            {
                "person_feat": torch.randn(1, N_PERSONS, FEAT_DIM, device=device),
                "person_mask": torch.ones(1, N_PERSONS, dtype=torch.bool, device=device),
                "person_frame": frames.view(1, N_PERSONS),
                "person_lat": torch.zeros(1, N_PERSONS, device=device),
                "person_lon": torch.zeros(1, N_PERSONS, device=device),
                "person_fov": torch.full((1, N_PERSONS), 45.0, device=device),
                "person_conf": torch.ones(1, N_PERSONS, device=device),
            }
        )
    return batch


def measure_module(name: str, module: nn.Module, example, forward) -> dict:
    module.eval()
    gmacs = gflops(module, example)
    milliseconds = latency_ms(autocast_call(forward))
    row = {
        "name": name,
        "params_m": param_count(module) / 1e6,
        "gmacs": gmacs,
        "latency_ms": milliseconds,
    }
    print(
        f"{name:22}  params {row['params_m']:.3f} M  gmacs {row['gmacs']:.3f} G  latency {row['latency_ms']:.2f} ms",
        flush=True,
    )
    return row


def check_sdpa_handle() -> None:
    """A masked encoder layer must include the two attention matmuls."""

    class MaskedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    256, 4, 1024, 0.1, activation="gelu", batch_first=True, norm_first=True
                ),
                num_layers=1,
                enable_nested_tensor=False,
            )

        def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.enc(tokens, src_key_padding_mask=mask)

    layer = MaskedEncoder().eval()
    tokens = torch.randn(1, 17, 256)
    mask = torch.zeros(1, 17, dtype=torch.bool)
    with torch.no_grad():
        total = flop_analysis(layer, (tokens, mask)).total()
    # Q, K, V, out, and both MLP projections, plus the QK^T and AV matmuls.
    linears = 4 * 17 * 256 * 256 + 2 * 17 * 256 * 1024
    attention = 2 * 1 * 4 * 17 * 17 * 64
    if total < linears + attention:
        raise RuntimeError(f"SDPA handle under-counted: total {total} < {linears + attention}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    fieldnames = ["name", "params_m", "gmacs", "latency_ms", "formula"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "params_m": f"{row['params_m']:.6f}",
                    "gmacs": f"{row['gmacs']:.6f}",
                    "latency_ms": f"{row['latency_ms']:.6f}",
                    "formula": row.get("formula", ""),
                }
            )
    os.replace(temporary, path)


def print_table(rows: list[dict]) -> None:
    headers = ("name", "params_M", "GMACs", "latency_ms", "formula")
    body = [
        (
            row["name"],
            f"{row['params_m']:.3f}",
            f"{row['gmacs']:.3f}",
            f"{row['latency_ms']:.2f}",
            row.get("formula", ""),
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for record in body:
        for index, cell in enumerate(record):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)), flush=True)
    print("  ".join("-" * width for width in widths), flush=True)
    for record in body:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(record)), flush=True)


def clip_total(name: str, formula: str, terms: list[tuple[dict, float]], param_rows: list[dict]) -> dict:
    return {
        "name": name,
        "params_m": sum(row["params_m"] for row in param_rows),
        "gmacs": sum(multiplier * row["gmacs"] for row, multiplier in terms),
        "latency_ms": sum(multiplier * row["latency_ms"] for row, multiplier in terms),
        "formula": formula,
    }


def main() -> None:
    logging.getLogger("fvcore").setLevel(logging.ERROR)
    args = parse_args()
    os.chdir(REPO_ROOT)
    device = torch.device(f"cuda:{args.gpu}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    check_sdpa_handle()
    configs = load_configs()
    rows = []

    print(f"device cuda:{args.gpu} {torch.cuda.get_device_name(device)}", flush=True)
    warn_if_gpu_busy(args.gpu)
    dino = load_dinov2().to(device).eval()
    dino_wrap = DinoForward(dino).eval()
    shapes = {
        "A": (1, 3, 448, 896),
        "B": (6, 3, 224, 224),
        "C": (1, 3, 224, 224),
        "C75": (N_PERSONS, 3, 224, 224),
    }
    inputs = {key: torch.randn(*shape, device=device) for key, shape in shapes.items()}
    labels = {
        "A": "A dinov2 896x448",
        "B": "B dinov2 6x224",
        "C": "C dinov2 224",
        "C75": "C75 dinov2 75x224",
    }
    measured = {}
    with torch.no_grad():
        for key in ("A", "B", "C", "C75"):
            pixel_values = inputs[key]
            measured[key] = measure_module(
                labels[key],
                dino_wrap,
                (pixel_values,),
                lambda pixel_values=pixel_values: dino_wrap(pixel_values),
            )
            rows.append(measured[key])
    del dino_wrap, dino, inputs
    torch.cuda.empty_cache()

    from ultralytics import YOLO

    yolo_specs = (
        ("D", "D yolo11x 2560x1024", "yolo11x.pt", (1024, 2560)),
        ("E", "E yolo11s 1280x512", "yolo11s.pt", (512, 1280)),
    )
    with torch.no_grad():
        for key, label, weight, image_hw in yolo_specs:
            detector = YOLO(str(REPO_ROOT / weight) if (REPO_ROOT / weight).is_file() else weight)
            network = detector.model.to(device).eval()
            height, width = image_hw
            image = torch.randn(1, 3, height, width, device=device)
            params, gmacs = yolo_gmacs(network, image_hw)
            milliseconds = latency_ms(autocast_call(lambda image=image, network=network: network(image)))
            row = {"name": label, "params_m": params / 1e6, "gmacs": gmacs, "latency_ms": milliseconds}
            measured[key] = row
            rows.append(row)
            print(
                f"{label:22}  params {row['params_m']:.3f} M  gmacs {row['gmacs']:.3f} G  latency {row['latency_ms']:.2f} ms",
                flush=True,
            )
            del network, detector, image
            torch.cuda.empty_cache()

    with torch.no_grad():
        for config_name in HEAD_CONFIGS:
            model = HARModel(configs[config_name]).to(device).eval()
            example = head_example(configs[config_name], device)
            keys = tuple(example)
            wrapped = TensorCall(model, keys).eval()
            tensors = tuple(example[key] for key in keys)
            label = f"F {config_name}"
            measured[label] = measure_module(
                label,
                wrapped,
                tensors,
                lambda wrapped=wrapped, tensors=tensors: wrapped(*tensors),
            )
            rows.append(measured[label])
            del wrapped, model, example
            torch.cuda.empty_cache()

    head = {name: measured[f"F {name}"] for name in HEAD_CONFIGS}
    clips = [
        clip_total("erp_cls_mean", "16A+F", [(measured["A"], 16), (head["erp_cls_mean"], 1)], [measured["A"], head["erp_cls_mean"]]),
        clip_total("cubemap", "16B+F", [(measured["B"], 16), (head["cubemap"], 1)], [measured["A"], head["cubemap"]]),
        clip_total("grid_latlon", "16A+F", [(measured["A"], 16), (head["grid_latlon"], 1)], [measured["A"], head["grid_latlon"]]),
        clip_total(
            "full",
            "16A+16D+C75+F",
            [(measured["A"], 16), (measured["D"], 16), (measured["C75"], 1), (head["full"], 1)],
            [measured["A"], measured["D"], head["full"]],
        ),
        clip_total(
            "full_light",
            "16A+16E+C75+F",
            [(measured["A"], 16), (measured["E"], 16), (measured["C75"], 1), (head["full"], 1)],
            [measured["A"], measured["E"], head["full"]],
        ),
    ]
    rows.extend(clips)
    print(flush=True)
    print_table(rows)
    print(
        "Clip params add each network once (A, C, and C75 share DINOv2). "
        "Clip GMACs and latency follow the formula (C75 is one batched call). "
        "fvcore GMACs are multiply counts; YOLO model.info FLOPs are divided by 2.",
        flush=True,
    )
    write_csv(RESULTS_PATH, rows)
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
