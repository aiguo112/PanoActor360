#!/usr/bin/env python3
"""Write IEEE tables and result macros from cached experiment outputs.

Reads results/{config}/summary.json, results/subset_eval.csv, results/cost.csv,
results/full/preds_fold{0-4}.json, and the protocol CSV. Writes
paper_fg2027/tables/{main,perclass,confusions,ablation,subsets,cost}.tex and
paper_fg2027/sections/numbers.tex.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
PAPER_ROOT = REPO_ROOT / "paper_fg2027"
# Author tree uses paper_fg2027/; code-only release writes into paper_tables/.
if (PAPER_ROOT / "tables").is_dir():
    PAPER_TABLES = PAPER_ROOT / "tables"
    NUMBERS_PATH = PAPER_ROOT / "sections" / "numbers.tex"
else:
    PAPER_TABLES = REPO_ROOT / "paper_tables"
    NUMBERS_PATH = PAPER_TABLES / "numbers.tex"
PROTOCOL_CSV = REPO_ROOT / "outputs" / "protocol" / "segments_v1.csv"
CLASSES_PATH = REPO_ROOT / "outputs" / "protocol" / "classes_v1.json"
# Release / code packs may ship protocol under protocol/ instead of outputs/protocol/.
if not PROTOCOL_CSV.is_file():
    PROTOCOL_CSV = REPO_ROOT / "protocol" / "segments_v1.csv"
if not CLASSES_PATH.is_file():
    CLASSES_PATH = REPO_ROOT / "protocol" / "classes_v1.json"
FRAMES_ROOT = REPO_ROOT / "data" / "frames_2048"
N_CLASSES = 27
N_FOLDS = 5

MAIN_CONFIGS = (
    "erp_cls_mean",
    "erp_cls_temporal",
    "cubemap",
    "grid_plain",
    "grid_latlon",
    "persons_only",
    "full",
    "full_light",
)
# External baselines appended after a midrule in Table I.
EXTERNAL_MAIN = (
    "videomae",
    "clip_person",
    "siglip_person",
)
ABLATION_CONFIGS = (
    "full",
    "abl_no_lat",
    "abl_lin_lon",
    "abl_no_area",
    "abl_naive",
    "abl_no_mil",
    "abl_no_roll",
)
SUBSET_CONFIGS = (
    "full",
    "erp_cls_mean",
    "erp_cls_temporal",
    "cubemap",
    "grid_plain",
    "grid_latlon",
    "persons_only",
    "full_light",
    "abl_no_lat",
    "abl_lin_lon",
    "abl_no_area",
    "abl_naive",
    "abl_no_mil",
    "abl_no_roll",
)
SUBSET_NAMES = ("SEAM", "NO_SEAM", "NEAR", "FAR", "NONE_DETECTED")
COST_CONFIGS = (
    "erp_cls_mean",
    "cubemap",
    "grid_latlon",
    "full",
    "full_light",
)
DISPLAY = {
    "erp_cls_mean": "ERP-CLS (mean)",
    "erp_cls_temporal": r"ERP-CLS + temporal",
    "cubemap": "Cubemap",
    "grid_plain": "ERP grid",
    "grid_latlon": r"ERP grid + spherical",
    "persons_only": "Person stream only",
    "full": "PanoActor (ours)",
    "full_light": "PanoActor-Lite (ours)",
    "videomae": r"VideoMAE (frozen)",
    "clip_person": r"CLIP zs (person)",
    "siglip_person": r"SigLIP zs (person)",
    "abl_no_lat": r"w/o latitude emb.",
    "abl_lin_lon": "linear longitude",
    "abl_no_area": r"w/o area pooling",
    "abl_naive": "naive ERP crops",
    "abl_no_mil": r"w/o actor attention",
    "abl_no_roll": r"w/o yaw roll",
}
ZEROSHOT_KEYS = {
    "clip_person": ("clip", "person"),
    "siglip_person": ("siglip", "person"),
}
TRACKED_CONFIGS = tuple(
    dict.fromkeys(MAIN_CONFIGS + ABLATION_CONFIGS + SUBSET_CONFIGS + COST_CONFIGS)
)


def fold_path(config: str, fold: int) -> Path:
    return RESULTS_ROOT / config / f"fold{fold}.json"


def fold_count(config: str) -> int:
    return sum(fold_path(config, fold).is_file() for fold in range(N_FOLDS))


def is_complete(config: str) -> bool:
    return fold_count(config) == N_FOLDS


def load_fold(config: str, fold: int) -> dict | None:
    path = fold_path(config, fold)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(config: str) -> dict | None:
    path = RESULTS_ROOT / config / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_zeroshot_rows() -> dict[tuple[str, str], dict]:
    path = RESULTS_ROOT / "zeroshot.csv"
    if not path.is_file():
        return {}
    by_key: dict[tuple[str, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_key[(row["model"], row["view"])] = {
                "top1": float(row["top1_mean"]),
                "top1_std": float(row["top1_std"]),
                "top5": None,
                "top5_std": None,
                "mca": float(row["mca_mean"]),
                "mca_std": float(row["mca_std"]),
            }
    return by_key


def external_metric_rows() -> list[dict]:
    rows: list[dict] = []
    if is_complete("videomae"):
        summary = load_summary("videomae")
        if summary is not None:
            rows.append(
                {
                    "config": "videomae",
                    "top1": float(summary["top1_mean"]),
                    "top1_std": float(summary.get("top1_std", 0.0)),
                    "top5": float(summary["top5_mean"]),
                    "top5_std": float(summary.get("top5_std", 0.0)),
                    "mca": float(summary["mca_mean"]),
                    "mca_std": float(summary.get("mca_std", 0.0)),
                }
            )
    zeroshot = load_zeroshot_rows()
    for config, key in ZEROSHOT_KEYS.items():
        payload = zeroshot.get(key)
        if payload is None:
            continue
        rows.append({"config": config, **payload})
    return rows


def display_name(config: str) -> str:
    return DISPLAY.get(config, config.replace("_", r"\_"))


def write_tex(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def best_indices(values: list[float | None], higher: bool) -> set[int]:
    ranked = [(index, value) for index, value in enumerate(values) if value is not None]
    if not ranked:
        return set()
    target = max(item[1] for item in ranked) if higher else min(item[1] for item in ranked)
    return {index for index, value in ranked if abs(value - target) <= 1e-9}


def mean_std(mean: float | None, std: float | None, bold: bool) -> str:
    if mean is None:
        return "---"
    if std is None:
        text = f"{mean:.2f}"
    else:
        text = f"{mean:.2f} $\\pm$ {std:.2f}"
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def fmt_num(value: float | None, digits: int, bold: bool = False, empty: str = "---") -> str:
    if value is None:
        return empty
    text = f"{value:.{digits}f}"
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "---"
    if abs(value) < 5e-4:
        return "0.00"
    return f"{value:+.2f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def incomplete_configs(configs: tuple[str, ...]) -> list[str]:
    return [config for config in configs if not is_complete(config)]


def metric_rows(configs: tuple[str, ...]) -> list[dict]:
    rows = []
    for config in configs:
        if not is_complete(config):
            continue
        summary = load_summary(config)
        if summary is None:
            continue
        rows.append(
            {
                "config": config,
                "top1": float(summary["top1_mean"]),
                "top1_std": float(summary.get("top1_std", 0.0)),
                "top5": float(summary["top5_mean"]),
                "top5_std": float(summary.get("top5_std", 0.0)),
                "mca": float(summary["mca_mean"]),
                "mca_std": float(summary.get("mca_std", 0.0)),
            }
        )
    return rows


def evaluated_clips(config: str) -> int | None:
    if not is_complete(config):
        return None
    total = 0
    for fold in range(N_FOLDS):
        payload = load_fold(config, fold)
        if payload is None or "n" not in payload:
            return None
        total += int(payload["n"])
    return total


def main_table(rows: list[dict], external: list[dict] | None = None) -> str:
    external = external or []
    all_rows = list(rows) + list(external)
    top1_best = best_indices([row["top1"] for row in all_rows], higher=True)
    top5_best = best_indices([row["top5"] for row in all_rows], higher=True)
    mca_best = best_indices([row["mca"] for row in all_rows], higher=True)
    cube_n = evaluated_clips("cubemap")
    caption = (
        r"Five-fold test accuracy (mean $\pm$ std, \%). "
        r"Rows above the rule share frozen DINOv2 features; "
        r"rows below are external video / zero-shot baselines."
    )
    if cube_n is not None and cube_n != 4259:
        cube_tex = f"{cube_n:,}".replace(",", "{,}")
        caption += (
            f" Cubemap is scored on the {cube_tex} clips with cached face features; "
            "every other row uses all 4{,}259 clips."
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:main}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & top-1 & top-5 & MCA \\",
        r"\midrule",
    ]
    if not all_rows:
        lines.append(r"\multicolumn{4}{c}{---} \\")
    for index, row in enumerate(all_rows):
        if index == len(rows) and rows and external:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                (
                    display_name(row["config"]),
                    mean_std(row["top1"], row["top1_std"], index in top1_best),
                    mean_std(row["top5"], row["top5_std"], index in top5_best),
                    mean_std(row["mca"], row["mca_std"], index in mca_best),
                )
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def ablation_table(rows: list[dict]) -> str:
    base = next((row for row in rows if row["config"] == "full"), None)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablations of PanoActor. $\Delta$ is the difference to PanoActor.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & top-1 & $\Delta$ top-1 & MCA & $\Delta$ MCA \\",
        r"\midrule",
    ]
    if not rows:
        lines.append(r"\multicolumn{5}{c}{---} \\")
    for row in rows:
        d_top = None if base is None else row["top1"] - base["top1"]
        d_mca = None if base is None else row["mca"] - base["mca"]
        name = "PanoActor" if row["config"] == "full" else display_name(row["config"])
        lines.append(
            " & ".join(
                (
                    name,
                    mean_std(row["top1"], row["top1_std"], False),
                    fmt_delta(d_top),
                    mean_std(row["mca"], row["mca_std"], False),
                    fmt_delta(d_mca),
                )
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def load_subset_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            def number(key: str) -> float | None:
                text = (row.get(key) or "").strip()
                if text == "":
                    return None
                return float(text)

            model = (row.get("model") or row.get("config") or "").strip()
            rows.append(
                {
                    "config": model,
                    "subset": row["subset"],
                    "n": int(float(row["n"])) if (row.get("n") or "").strip() else 0,
                    "top1": number("top1_mean") if "top1_mean" in row else number("top1"),
                    "top1_std": number("top1_std"),
                    "diff_mean": number("delta_vs_panoactor") if "delta_vs_panoactor" in row else number("diff_mean"),
                    "p_value": number("p_ttest") if "p_ttest" in row else number("p_value"),
                    "p_pooled": number("p_pooled"),
                }
            )
    return rows


def subset_cell(row: dict | None) -> str:
    if row is None or row["top1"] is None:
        return "---"
    if row["config"] == "full":
        return fmt_num(row["top1"], 2)
    delta = fmt_delta(row["diff_mean"])
    mark = ""
    pooled = row.get("p_pooled")
    if pooled is not None and pooled < 0.05:
        mark = r"$^{\dagger}$"
    return f"{fmt_num(row['top1'], 2)} ({delta}){mark}"


def subset_table(rows: list[dict]) -> str:
    by_key = {(row["config"], row["subset"]): row for row in rows}
    present = [
        config
        for config in SUBSET_CONFIGS
        if is_complete(config) and any((config, subset) in by_key for subset in SUBSET_NAMES)
    ]
    counts = {}
    for subset in SUBSET_NAMES:
        full = by_key.get(("full", subset))
        counts[subset] = None if full is None else full["n"]
    header_bits = []
    for subset, title in (
        ("SEAM", "Seam"),
        ("NO_SEAM", "No-seam"),
        ("NEAR", "Near"),
        ("FAR", "Far"),
        ("NONE_DETECTED", "None"),
    ):
        count = counts[subset]
        count_tex = "---" if count is None else f"{count:,}".replace(",", "{,}")
        header_bits.append(f"{title} ($n{{=}}{count_tex}$)")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Subset top-1 (mean over test folds, \%). Parentheses are the paired per-fold difference to PanoActor. A dagger marks an exact McNemar $p<0.05$ on pooled clips.}",
        r"\label{tab:subsets}",
        r"{\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        " & " + " & ".join(header_bits) + r" \\",
        r"\midrule",
    ]
    if not present:
        lines.append(r"\multicolumn{6}{c}{---} \\")
    for config in present:
        name = "PanoActor" if config == "full" else display_name(config)
        cells = [subset_cell(by_key.get((config, subset))) for subset in SUBSET_NAMES]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def load_cost_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    by_name: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            compute_key = "gmacs" if (row.get("gmacs") or "").strip() else "flops_g"
            compute = (row.get(compute_key) or "").strip()
            params = (row.get("params_m") or "").strip()
            latency = (row.get("latency_ms") or "").strip()
            by_name[row["name"]] = {
                "params_m": float(params) if params else None,
                "gmacs": float(compute) if compute else None,
                "latency_ms": float(latency) if latency else None,
                "formula": (row.get("formula") or "").strip(),
            }
    return by_name


def cost_table(by_name: dict[str, dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-clip compute on one RTX 3060. Trained parameters exclude the frozen encoders.}",
        r"\label{tab:cost}",
        r"{\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" & trained params (M) & total params (M) & GMACs/clip & latency/clip (ms) & top-1 \\",
        r"\midrule",
    ]
    wrote = False
    for config in COST_CONFIGS:
        if not is_complete(config):
            continue
        clip = by_name.get(config)
        if clip is None or not clip["formula"]:
            continue
        wrote = True
        head = by_name.get(f"F {config}")
        if head is None and config == "full_light":
            head = by_name.get("F full")
        trained = None if head is None else head["params_m"]
        summary = load_summary(config)
        top1 = None if summary is None else float(summary["top1_mean"])
        name = "PanoActor" if config == "full" else display_name(config)
        if config == "full_light":
            name = "PanoActor-Lite"
        lines.append(
            " & ".join(
                (
                    name,
                    fmt_num(trained, 3),
                    fmt_num(clip["params_m"], 3),
                    fmt_num(clip["gmacs"], 1),
                    fmt_num(clip["latency_ms"], 1),
                    fmt_num(top1, 2),
                )
            )
            + r" \\"
        )
    if not wrote:
        lines.append(r"\multicolumn{6}{c}{---} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def load_protocol_rows() -> tuple[dict[str, str], list[dict]]:
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    have_frames = {path.name for path in FRAMES_ROOT.iterdir() if path.is_dir()}
    rows = []
    with PROTOCOL_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["seg_id"] in have_frames:
                rows.append(row)
    return classes, rows


def perclass_stats(classes: dict[str, str], rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_idx"])].append(row)
    acc_folds: dict[int, list[float]] = defaultdict(list)
    for fold in range(5):
        path = RESULTS_ROOT / "full" / f"fold{fold}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.get("per_class_acc", {}).items():
            acc_folds[int(key)].append(float(value))
    stats = []
    for index in range(N_CLASSES):
        items = grouped[index]
        accs = acc_folds.get(index, [])
        stats.append(
            {
                "name": classes[str(index)],
                "clips": len(items),
                "videos": len({item["video_id"] for item in items}),
                "duration": float(np.mean([float(item["duration"]) for item in items])) if items else 0.0,
                "acc": float(np.mean(accs)) if accs else None,
            }
        )
    stats.sort(key=lambda item: (-item["clips"], item["name"]))
    return stats


def perclass_table(stats: list[dict]) -> str:
    mid = (len(stats) + 1) // 2

    def half(rows: list[dict]) -> list[str]:
        lines = [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"class & clips & videos & dur (s) & top-1 \\",
            r"\midrule",
        ]
        for row in rows:
            acc = "---" if row["acc"] is None else f"{row['acc']:.1f}"
            lines.append(
                f"{row['name']} & {row['clips']} & {row['videos']} & {row['duration']:.1f} & {acc} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return lines

    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Per-class inventory on the 4{,}259-clip protocol and full-model top-1, averaged over test folds that contain the class. Duration is the mean segment length.}",
            r"\label{tab:perclass}",
            r"{\small",
            r"\begin{minipage}[t]{0.48\textwidth}",
            r"\centering",
            *half(stats[:mid]),
            r"\end{minipage}",
            r"\hfill",
            r"\begin{minipage}[t]{0.48\textwidth}",
            r"\centering",
            *half(stats[mid:]),
            r"\end{minipage}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def load_all_full_preds() -> list[dict]:
    records = []
    for fold in range(5):
        path = RESULTS_ROOT / "full" / f"preds_fold{fold}.json"
        if not path.is_file():
            print(f"missing {path}", flush=True)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload)
    return records


def confusions_table(classes: dict[str, str], records: list[dict]) -> str:
    counts: Counter[tuple[int, int]] = Counter()
    for record in records:
        true = int(record["true"])
        pred = int(record["pred"])
        if true != pred:
            counts[(true, pred)] += 1
    ranked = counts.most_common(8)
    n_clips = len(records)
    n_tex = f"{n_clips:,}".replace(",", "{,}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Most frequent full-model confusions, pooled over all test folds ($n{=}"
        + n_tex
        + r"$ clips).}",
        r"\label{tab:confusions}",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"true & predicted & $n$ \\",
        r"\midrule",
    ]
    if not ranked:
        lines.append(r"\multicolumn{3}{c}{---} \\")
    for (true, pred), count in ranked:
        lines.append(f"{classes[str(true)]} & {classes[str(pred)]} & {count} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    print(f"confusions from {n_clips} clips", flush=True)
    return "\n".join(lines)


def dec1(value: float) -> str:
    return f"{value:.1f}"


def cost_field(name: str, field: str) -> float | None:
    row = load_cost_rows(RESULTS_ROOT / "cost.csv").get(name)
    if row is None:
        return None
    value = row.get(field)
    return None if value is None else float(value)


def clip_gmacs(config: str) -> float | None:
    rows = load_cost_rows(RESULTS_ROOT / "cost.csv")
    clip = rows.get(config)
    if clip is None or not clip["formula"] or clip["gmacs"] is None:
        return None
    return float(clip["gmacs"])


def n_folds_win() -> int | None:
    if not all(is_complete(name) for name in ("full", "erp_cls_mean", "cubemap")):
        return None
    wins = 0
    for fold in range(N_FOLDS):
        full = load_fold("full", fold)
        erp = load_fold("erp_cls_mean", fold)
        cube = load_fold("cubemap", fold)
        if full is None or erp is None or cube is None:
            return None
        if float(full["top1"]) > float(erp["top1"]) and float(full["top1"]) > float(cube["top1"]):
            wins += 1
    return wins


def numbers_tex() -> str:
    required = {
        "full": load_summary("full") if is_complete("full") else None,
        "erp_cls_mean": load_summary("erp_cls_mean") if is_complete("erp_cls_mean") else None,
        "cubemap": load_summary("cubemap") if is_complete("cubemap") else None,
        "grid_plain": load_summary("grid_plain") if is_complete("grid_plain") else None,
        "grid_latlon": load_summary("grid_latlon") if is_complete("grid_latlon") else None,
        "abl_no_mil": load_summary("abl_no_mil") if is_complete("abl_no_mil") else None,
        "abl_no_lat": load_summary("abl_no_lat") if is_complete("abl_no_lat") else None,
        "abl_lin_lon": load_summary("abl_lin_lon") if is_complete("abl_lin_lon") else None,
    }
    missing = [name for name, payload in required.items() if payload is None]
    if missing:
        raise SystemExit(f"missing complete 5-fold summaries for: {', '.join(missing)}")
    full = required["full"]
    erp = required["erp_cls_mean"]
    cube = required["cubemap"]
    grid_plain = required["grid_plain"]
    grid = required["grid_latlon"]
    no_mil = required["abl_no_mil"]
    no_lat = required["abl_no_lat"]
    lin_lon = required["abl_lin_lon"]
    acc_full = float(full["top1_mean"])
    mca_full = float(full["mca_mean"])
    acc_erp = float(erp["top1_mean"])
    acc_cube = float(cube["top1_mean"])
    mca_erp = float(erp["mca_mean"])
    acc_grid = float(grid["top1_mean"])
    mca_grid = float(grid["mca_mean"])
    lines = [
        r"% Auto-generated by scripts/make_tables.py -- do not edit by hand.",
        r"\newcommand{\nClips}{4{,}259}",
        r"\newcommand{\nVideos}{231}",
        r"\newcommand{\nClasses}{27}",
        r"\newcommand{\pctSeam}{8.9}",
        r"\newcommand{\pctCrowd}{35}",
        rf"\newcommand{{\accFull}}{{{dec1(acc_full)}}}",
        rf"\newcommand{{\mcaFull}}{{{dec1(mca_full)}}}",
        rf"\newcommand{{\accERP}}{{{dec1(acc_erp)}}}",
        rf"\newcommand{{\accCube}}{{{dec1(acc_cube)}}}",
        rf"\newcommand{{\gainERP}}{{{dec1(acc_full - acc_erp)}}}",
        rf"\newcommand{{\gainCube}}{{{dec1(acc_full - acc_cube)}}}",
        rf"\newcommand{{\gainMcaERP}}{{{dec1(mca_full - mca_erp)}}}",
        rf"\newcommand{{\gainGrid}}{{{dec1(float(grid_plain['top1_mean']) - acc_erp)}}}",
        rf"\newcommand{{\gainPerson}}{{{dec1(acc_full - acc_grid)}}}",
        rf"\newcommand{{\gainMcaPerson}}{{{dec1(mca_full - mca_grid)}}}",
        rf"\newcommand{{\accNoMil}}{{{dec1(float(no_mil['top1_mean']))}}}",
        rf"\newcommand{{\mcaNoMil}}{{{dec1(float(no_mil['mca_mean']))}}}",
        rf"\newcommand{{\dMcaNoLat}}{{{dec1(mca_full - float(no_lat['mca_mean']))}}}",
        rf"\newcommand{{\dMcaLinLon}}{{{dec1(mca_full - float(lin_lon['mca_mean']))}}}",
    ]
    wins = n_folds_win()
    if wins is not None:
        lines.append(rf"\newcommand{{\nFoldsWin}}{{{wins}}}")
    persons = load_summary("persons_only") if is_complete("persons_only") else None
    if persons is not None:
        lines.append(rf"\newcommand{{\accPersons}}{{{dec1(float(persons['top1_mean']))}}}")
        lines.append(rf"\newcommand{{\mcaPersons}}{{{dec1(float(persons['mca_mean']))}}}")
    light = load_summary("full_light") if is_complete("full_light") else None
    if light is not None:
        lines.append(rf"\newcommand{{\accLight}}{{{dec1(float(light['top1_mean']))}}}")
        lines.append(rf"\newcommand{{\mcaLight}}{{{dec1(float(light['mca_mean']))}}}")
    videomae = load_summary("videomae") if is_complete("videomae") else None
    if videomae is not None:
        acc_vm = float(videomae["top1_mean"])
        lines.append(rf"\newcommand{{\accVideoMAE}}{{{dec1(acc_vm)}}}")
        lines.append(rf"\newcommand{{\mcaVideoMAE}}{{{dec1(float(videomae['mca_mean']))}}}")
        lines.append(rf"\newcommand{{\gainVideoMAE}}{{{dec1(acc_full - acc_vm)}}}")
    zeroshot = load_zeroshot_rows()
    clip_person = zeroshot.get(("clip", "person"))
    clip_erp = zeroshot.get(("clip", "erp"))
    siglip_person = zeroshot.get(("siglip", "person"))
    if clip_person is not None:
        lines.append(rf"\newcommand{{\accCLIP}}{{{dec1(clip_person['top1'])}}}")
        lines.append(rf"\newcommand{{\mcaCLIP}}{{{dec1(clip_person['mca'])}}}")
    if clip_erp is not None:
        lines.append(rf"\newcommand{{\accCLIPERP}}{{{dec1(clip_erp['top1'])}}}")
    if siglip_person is not None:
        lines.append(rf"\newcommand{{\accSigLIP}}{{{dec1(siglip_person['top1'])}}}")
        lines.append(rf"\newcommand{{\mcaSigLIP}}{{{dec1(siglip_person['mca'])}}}")
    erp_gmacs = clip_gmacs("erp_cls_mean")
    full_gmacs = clip_gmacs("full")
    light_gmacs = clip_gmacs("full_light")
    erp_latency = cost_field("erp_cls_mean", "latency_ms")
    full_latency = cost_field("full", "latency_ms")
    light_latency = cost_field("full_light", "latency_ms")
    detector_latency = cost_field("D yolo11x 2560x1024", "latency_ms")
    crop_latency = cost_field("C dinov2 224", "latency_ms")
    if erp_gmacs and full_gmacs:
        lines.append(rf"\newcommand{{\costRatioFull}}{{{dec1(full_gmacs / erp_gmacs)}}}")
        lines.append(rf"\newcommand{{\macRatio}}{{{dec1(full_gmacs / erp_gmacs)}}}")
    if erp_latency and full_latency:
        lines.append(rf"\newcommand{{\latRatio}}{{{dec1(full_latency / erp_latency)}}}")
    if erp_gmacs and light_gmacs:
        lines.append(rf"\newcommand{{\macRatioLight}}{{{dec1(light_gmacs / erp_gmacs)}}}")
        lines.append(rf"\newcommand{{\costRatioLight}}{{{dec1(light_gmacs / erp_gmacs)}}}")
    if erp_latency and light_latency:
        lines.append(rf"\newcommand{{\latRatioLight}}{{{dec1(light_latency / erp_latency)}}}")
    if detector_latency is not None:
        lines.append(rf"\newcommand{{\latDetector}}{{{dec1(16.0 * detector_latency)}}}")
    if crop_latency is not None:
        lines.append(rf"\newcommand{{\latCrops}}{{{dec1(75.0 * crop_latency)}}}")
    if persons is not None:
        lines.append(rf"\newcommand{{\gapPersonsTop}}{{{dec1(acc_full - float(persons['top1_mean']))}}}")
        lines.append(rf"\newcommand{{\gapPersonsMca}}{{{dec1(mca_full - float(persons['mca_mean']))}}}")
    if erp_gmacs:
        lines.append(rf"\newcommand{{\macERP}}{{{erp_gmacs / 1000.0:.2f}}}")
    counts_path = RESULTS_ROOT / "subsets" / "counts.csv"
    count_macros = {
        "SEAM": "nSeam",
        "NO_SEAM": "nNoSeam",
        "NEAR": "nNear",
        "FAR": "nFar",
        "NONE_DETECTED": "nNone",
    }
    if counts_path.is_file():
        with counts_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                macro = count_macros.get(row["subset"])
                if macro is not None:
                    lines.append(rf"\newcommand{{\{macro}}}{{{int(row['n'])}}}")
    for name in (
        "nFoldsWin",
        "accPersons",
        "mcaPersons",
        "gapPersonsTop",
        "gapPersonsMca",
        "accLight",
        "mcaLight",
        "accVideoMAE",
        "mcaVideoMAE",
        "gainVideoMAE",
        "accCLIP",
        "mcaCLIP",
        "accCLIPERP",
        "accSigLIP",
        "mcaSigLIP",
        "costRatioFull",
        "costRatioLight",
        "macRatio",
        "latRatio",
        "macRatioLight",
        "latRatioLight",
        "latDetector",
        "latCrops",
        "macERP",
        "nSeam",
        "nNoSeam",
        "nNear",
        "nFar",
        "nNone",
    ):
        lines.append(rf"\providecommand{{\{name}}}{{\textcolor{{red}}{{[{name}]}}}}")
    lines.append("")
    return "\n".join(lines)


def zeroshot_table() -> str:
    """Compact all-view zero-shot table for the paper body or supplement."""
    zeroshot = load_zeroshot_rows()
    order = (
        ("clip", "erp", r"CLIP / ERP"),
        ("clip", "cubemap", r"CLIP / cubemap"),
        ("clip", "person", r"CLIP / person"),
        ("clip", "person_erp", r"CLIP / person+ERP"),
        ("siglip", "erp", r"SigLIP / ERP"),
        ("siglip", "cubemap", r"SigLIP / cubemap"),
        ("siglip", "person", r"SigLIP / person"),
        ("siglip", "person_erp", r"SigLIP / person+ERP"),
    )
    top1_vals = [zeroshot[k]["top1"] for k in ((m, v) for m, v, _ in order) if k in zeroshot]
    mca_vals = [zeroshot[k]["mca"] for k in ((m, v) for m, v, _ in order) if k in zeroshot]
    top1_best = best_indices(top1_vals, higher=True) if top1_vals else set()
    mca_best = best_indices(mca_vals, higher=True) if mca_vals else set()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zero-shot CLIP and SigLIP on the same five folds (mean $\pm$ std, \%). Person crops use max-over-persons then mean-over-frames.}",
        r"\label{tab:zeroshot}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & top-1 & MCA \\",
        r"\midrule",
    ]
    if not zeroshot:
        lines.append(r"\multicolumn{3}{c}{---} \\")
    idx = 0
    for model, view, name in order:
        payload = zeroshot.get((model, view))
        if payload is None:
            continue
        lines.append(
            " & ".join(
                (
                    name,
                    mean_std(payload["top1"], payload["top1_std"], idx in top1_best),
                    mean_std(payload["mca"], payload["mca_std"], idx in mca_best),
                )
            )
            + r" \\"
        )
        idx += 1
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    missing = incomplete_configs(TRACKED_CONFIGS)
    if missing:
        listed = ", ".join(f"{name} ({fold_count(name)}/5)" for name in missing)
        print(f"warning: configs with <5 folds excluded from tables: {listed}", flush=True)
    classes, protocol_rows = load_protocol_rows()
    write_tex(
        PAPER_TABLES / "main.tex",
        main_table(metric_rows(MAIN_CONFIGS), external_metric_rows()),
    )
    write_tex(PAPER_TABLES / "zeroshot.tex", zeroshot_table())
    write_tex(PAPER_TABLES / "perclass.tex", perclass_table(perclass_stats(classes, protocol_rows)))
    write_tex(PAPER_TABLES / "confusions.tex", confusions_table(classes, load_all_full_preds()))
    write_tex(PAPER_TABLES / "ablation.tex", ablation_table(metric_rows(ABLATION_CONFIGS)))
    write_tex(PAPER_TABLES / "subsets.tex", subset_table(load_subset_rows(RESULTS_ROOT / "subset_eval.csv")))
    write_tex(PAPER_TABLES / "cost.tex", cost_table(load_cost_rows(RESULTS_ROOT / "cost.csv")))
    write_tex(NUMBERS_PATH, numbers_tex())
    print(
        f"wrote {PAPER_TABLES}/{{main,zeroshot,perclass,confusions,ablation,subsets,cost}}.tex",
        flush=True,
    )
    print(f"wrote {NUMBERS_PATH}", flush=True)


if __name__ == "__main__":
    main()
