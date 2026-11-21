# Research: automatic counting of chickens around feeders

Undergraduate research, National Taiwan University, Dept. of Biomechatronics Engineering (advisor Dr. Yan-Fu Kuo), presented 2025-05-20.

**Setup.** Commercial broiler house, 113 m × 14.8 m. A Raspberry Pi 4B with a 130° camera records 1920×1080 at 5 fps, and inference runs offline on a lab GPU server.

**Method.**
1. **Feeder detection.** YOLO11n @640, AP 97.5%. The camera is fixed, so this runs once.
2. **Perspective compensation.** 1-D K-means on feeder center y splits feeders into a far row and a near row. Far feeders look smaller, so each row gets its own distance threshold.
3. **Chicken detection.** YOLO11x @960, AP 91.6%. There are 650–750 birds per frame.
4. **"Around feeder" decision.** A chicken counts if the distance from its center to the nearest feeder is within that row's threshold. The thresholds (45.10 px far, 94.23 px near) are fitted per row from the distances of labelled eating chickens, after IQR outlier removal (`threshold.py`).

**Result.** Precision 61.6%, recall 61.7%, F1 0.617 for "around feeder".
Two failure modes: eating birds at the frame edge are missed (false negatives), and birds resting next to a feeder without eating are counted (false positives). Telling these apart was the original goal, **feeding-behavior detection**, and the project ended before reaching it.

| File | Purpose |
|---|---|
| `evaluate_images.py` | Feeder calibration (detect → K-means → `cached_feeders.json`) + per-image TP/FP/FN evaluation against labels |
| `threshold.py` | Fits the per-row distance thresholds from labelled data and plots the distance distributions |
| `eval_eating.py` | Hungarian-matched evaluation of the "around feeder" classification |
| `annotate_video.py` | Renders an annotated video with feeder rings and per-frame counts |
| `crop_feeder.py` | Crops feeder regions from video, used for dataset building |
| `analysis/daily_feeding_stats.py` | Full-day counting over 5-minute clips → CSV |
| `analysis/plot_feeding_stats.py` | Time-series plots from that CSV |
| `data.yaml`, `data_feeder.yaml` | Ultralytics dataset configs (chicken / feeder) |

These scripts are kept close to their original form: configuration lives in constants at the top of each file. Paths point to `data/` and `models/` relative to the repo root, and the dataset is not included. The cleaned, production version of this logic is in [`../pipeline/`](../pipeline/).
