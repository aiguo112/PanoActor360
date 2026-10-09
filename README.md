# PanoActor360

Person-centric action recognition in 360° panoramic (ERP) video.

PanoActor combines a global spherical stream with a seam-aware person stream (tangent crops + gated MIL attention) on frozen DINOv2 features.

## Repository contents

| Path | Contents |
| --- | --- |
| `panoactor/` | Model (fusion transformer + gated MIL) |
| `har/` | Import shim (`from har.model import ...`) |
| `scripts/` | Feature cache, train/eval, baselines, table export, demo render |
| `protocol/` | Five-fold video-grouped splits (`segments_v1.csv`) and class list |
| `results/` | Fold metrics, predictions, cost / subset / zeroshot summaries |
| `supplement/` | Face-blurred qualitative demo video + clip manifest |
| `configs/` | Local path placeholders |

## Not included

- Paper LaTeX / PDF
- Raw 360+x videos, extracted frames, or feature caches
- Checkpoints

Obtain **360+x** from its official release, then edit `configs/paths.yaml`.

## Reproduce summary numbers from shipped results

```bash
python scripts/make_tables.py
# writes paper_tables/*.tex and paper_tables/numbers.tex
```

## Retrain (after local feature extraction)

```bash
# 1) extract frames from your 360+x copy
# 2) detect + cache features (scripts/detect_and_crop.py, cache_features.py, pool_global.py)
# 3) train
python scripts/train_har.py --config full --folds 0,1,2,3,4
python scripts/train_har.py --config full_light --folds 0,1,2,3,4
python scripts/videomae_baseline.py
python scripts/zeroshot_eval.py
```

Protocol: 5 video-grouped folds, seed 42, 16 frames per clip.

## Demo video

- `supplement/panoactor_demo.mp4`
- `supplement/panoactor_demo.json`

Faces are blurred. Do not redistribute raw 360+x content.
