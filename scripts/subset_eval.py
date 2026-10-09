#!/usr/bin/env python3
"""Score saved test predictions on seam, distance, and latitude subsets.

Reads results/{config}/preds_fold{f}.json and data/person_meta/{seg_id}.json.
Person latitude in the meta files is radians and is converted to degrees.
Prints one table and writes results/subset_eval.csv. Does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
META_ROOT = REPO_ROOT / "data" / "person_meta"
N_FOLDS = 5
SUBSETS = ("ALL", "SEAM", "NO_SEAM", "NEAR", "FAR", "NONE_DETECTED")
SUBSET_DIR = REPO_ROOT / "results" / "subsets"
DEFAULT_CONFIGS = (
    "full",
    "full_light",
    "abl_lin_lon",
    "abl_naive",
    "abl_no_lat",
    "abl_no_area",
    "grid_latlon",
    "grid_plain",
    "erp_cls_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--meta", type=Path, default=META_ROOT)
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "subset_eval.csv")
    return parser.parse_args()


def load_fold_preds(results_root: Path, config: str) -> dict[int, list[dict]]:
    directory = results_root / config
    folds = {}
    missing = []
    for fold in range(N_FOLDS):
        path = directory / f"preds_fold{fold}.json"
        if not path.is_file():
            missing.append(path)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"{path} must be a list of prediction records")
        folds[fold] = payload
    if missing:
        listed = "\n".join(str(path) for path in missing)
        raise SystemExit(f"missing prediction files for {config}:\n{listed}")
    return folds


def segment_geometry(meta_root: Path, seg_id: str) -> dict | None:
    path = meta_root / f"{seg_id}.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    fovs = []
    abs_lats = []
    has_seam = False
    for frame in document.get("frames") or []:
        for person in frame:
            if int(person.get("seam", 0)) == 1:
                has_seam = True
            fovs.append(float(person["fov"]))
            abs_lats.append(abs(math.degrees(float(person["lat"]))))
    if not fovs:
        return {"has_seam": False, "max_fov": None, "n_persons": 0}
    return {"has_seam": has_seam, "max_fov": max(fovs), "n_persons": len(fovs)}


def in_subset(subset: str, geometry: dict | None) -> bool:
    if subset == "ALL":
        return True
    if geometry is None:
        return False
    if subset == "SEAM":
        return bool(geometry["has_seam"])
    if subset == "NO_SEAM":
        return not geometry["has_seam"]
    if subset == "NEAR":
        return geometry["max_fov"] is not None and geometry["max_fov"] >= 60.0
    if subset == "FAR":
        return geometry["n_persons"] > 0 and geometry["max_fov"] < 30.0
    if subset == "NONE_DETECTED":
        return geometry["n_persons"] == 0
    raise ValueError(subset)


def top1_and_mca(records: list[dict]) -> tuple[float | None, float | None]:
    if not records:
        return None, None
    correct = [int(record["pred"]) == int(record["true"]) for record in records]
    top1 = 100.0 * sum(correct) / len(records)
    by_class: dict[int, list[bool]] = {}
    for record, hit in zip(records, correct):
        by_class.setdefault(int(record["true"]), []).append(hit)
    mca = 100.0 * sum(sum(hits) / len(hits) for hits in by_class.values()) / len(by_class)
    return top1, mca


def fold_top1(records: list[dict]) -> float | None:
    top1, _mca = top1_and_mca(records)
    return top1


def mcnemar_p(reference_hits: list[bool], other_hits: list[bool]) -> float | None:
    """Exact McNemar test on paired clip correctness. reference is PanoActor."""
    if len(reference_hits) != len(other_hits) or not reference_hits:
        return None
    only_ref = 0
    only_other = 0
    for ref_hit, other_hit in zip(reference_hits, other_hits):
        if ref_hit and not other_hit:
            only_ref += 1
        elif other_hit and not ref_hit:
            only_other += 1
    discordant = only_ref + only_other
    if discordant == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(only_other, discordant, 0.5, alternative="two-sided").pvalue)


def paired_ttest(config_scores: list[float], full_scores: list[float]) -> float | None:
    if len(config_scores) < 2 or len(config_scores) != len(full_scores):
        return None
    from scipy.stats import ttest_rel

    result = ttest_rel(config_scores, full_scores)
    p_value = float(result.pvalue)
    if math.isnan(p_value):
        return None
    return p_value


def write_subset_lists(geometry: dict[str, dict | None], full_folds: dict[int, list[dict]]) -> None:
    """Clip ids are taken from the PanoActor test predictions, one file per subset."""
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[str]] = {subset: [] for subset in SUBSETS}
    seen = set()
    for fold in range(N_FOLDS):
        for record in full_folds[fold]:
            seg_id = str(record["seg_id"])
            if seg_id in seen:
                continue
            seen.add(seg_id)
            stats = geometry.get(seg_id)
            for subset in SUBSETS:
                if in_subset(subset, stats):
                    grouped[subset].append(seg_id)
    counts_path = SUBSET_DIR / "counts.csv"
    temporary = counts_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subset", "n"])
        writer.writeheader()
        for subset in SUBSETS:
            ids = grouped[subset]
            list_path = SUBSET_DIR / f"{subset}.txt"
            list_tmp = list_path.with_suffix(".txt.tmp")
            list_tmp.write_text("".join(f"{seg_id}\n" for seg_id in ids), encoding="utf-8")
            os.replace(list_tmp, list_path)
            writer.writerow({"subset": subset, "n": len(ids)})
    os.replace(temporary, counts_path)
    print(f"wrote clip lists under {SUBSET_DIR}", flush=True)


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "subset",
        "n",
        "top1_mean",
        "top1_std",
        "delta_vs_panoactor",
        "p_ttest",
        "p_pooled",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(temporary, path)


def print_table(rows: list[dict]) -> None:
    headers = ("model", "subset", "n", "top1", "std", "delta", "p_t", "p_pool")
    body = []
    for row in rows:
        body.append(
            (
                row["model"],
                row["subset"],
                str(row["n"]),
                fmt(row["top1_mean"]),
                fmt(row["top1_std"]),
                fmt(row["delta_vs_panoactor"]),
                "" if row["p_ttest"] is None else f"{row['p_ttest']:.4g}",
                "" if row["p_pooled"] is None else f"{row['p_pooled']:.4g}",
            )
        )
    widths = [len(header) for header in headers]
    for record in body:
        for index, cell in enumerate(record):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    print(line, flush=True)
    print("  ".join("-" * width for width in widths), flush=True)
    for record in body:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(record)), flush=True)


def main() -> None:
    args = parse_args()
    if "full" not in args.configs:
        raise SystemExit("--configs must include full; paired differences are measured against it")
    predictions = {config: load_fold_preds(args.results, config) for config in args.configs}
    seg_ids = {
        str(record["seg_id"])
        for folds in predictions.values()
        for records in folds.values()
        for record in records
    }
    geometry = {seg_id: segment_geometry(args.meta, seg_id) for seg_id in seg_ids}
    missing_meta = sum(stats is None for stats in geometry.values())
    if missing_meta:
        print(f"person meta missing for {missing_meta} segments; they stay in ALL only", flush=True)

    full_folds = predictions["full"]
    write_subset_lists(geometry, full_folds)
    rows = []
    for config in args.configs:
        for subset in SUBSETS:
            config_fold_scores = []
            full_fold_scores = []
            ref_hits: list[bool] = []
            other_hits: list[bool] = []
            n_clips = 0
            for fold in range(N_FOLDS):
                config_records = [
                    record
                    for record in predictions[config][fold]
                    if in_subset(subset, geometry.get(str(record["seg_id"])))
                ]
                full_records = [
                    record
                    for record in full_folds[fold]
                    if in_subset(subset, geometry.get(str(record["seg_id"])))
                ]
                full_by_id = {str(record["seg_id"]): record for record in full_records}
                matched_config = []
                matched_full = []
                for record in config_records:
                    other = full_by_id.get(str(record["seg_id"]))
                    if other is None:
                        continue
                    matched_config.append(record)
                    matched_full.append(other)
                n_clips += len(matched_config)
                config_top1 = fold_top1(matched_config)
                full_top1 = fold_top1(matched_full)
                if config_top1 is None or full_top1 is None:
                    continue
                config_fold_scores.append(config_top1)
                full_fold_scores.append(full_top1)
                if config != "full":
                    ref_hits.extend(int(item["pred"]) == int(item["true"]) for item in matched_full)
                    other_hits.extend(int(item["pred"]) == int(item["true"]) for item in matched_config)
            if config_fold_scores:
                mean = sum(config_fold_scores) / len(config_fold_scores)
                if len(config_fold_scores) > 1:
                    var = sum((value - mean) ** 2 for value in config_fold_scores) / (len(config_fold_scores) - 1)
                    std = math.sqrt(var)
                else:
                    std = 0.0
                delta = sum(a - b for a, b in zip(config_fold_scores, full_fold_scores)) / len(config_fold_scores)
            else:
                mean = std = delta = None
            rows.append(
                {
                    "model": config,
                    "subset": subset,
                    "n": n_clips,
                    "top1_mean": mean,
                    "top1_std": std,
                    "delta_vs_panoactor": 0.0 if config == "full" else delta,
                    "p_ttest": None if config == "full" else paired_ttest(config_fold_scores, full_fold_scores),
                    "p_pooled": None if config == "full" else mcnemar_p(ref_hits, other_hits),
                }
            )
    print_table(rows)
    serializable = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, float):
                item[key] = f"{value:.6f}"
            elif value is None:
                item[key] = ""
        serializable.append(item)
    write_csv(args.out, serializable)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
