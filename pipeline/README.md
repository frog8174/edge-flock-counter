# Pipeline: full-day feeding time series

This pipeline runs YOLO11x on every frame of a full day of poultry-house video. For each frame it decides which chickens are around the fixed feeders. It then produces time-series CSVs and an interactive HTML report.

It was developed and validated on a Jetson Orin NX and also runs on x86 GPU servers. The optimization history is in [`../docs/optimization-log.md`](../docs/optimization-log.md).

```
one day of *.mp4
   │  daily_infer.py          per-frame YOLO + feeder-distance decision
   ▼
per-clip CSV (results/<date>/*.csv)
   │  timeseries_analysis.py  merge + bucket by time
   ▼
raw / 1-min / 5-min CSV (results/<date>_summary/)
   │  generate_report.py      statistics + template
   ▼
interactive HTML report (results/<date>_report.html)
```

Each stage hands off through CSV, so any step can be re-run or replaced on its own.

| Path | Purpose |
|---|---|
| `scripts/daily_infer.py` | Batch inference over a day of video → per-frame CSV. **Main entry point.** Resumable: clips that already have a CSV are skipped. |
| `scripts/timeseries_analysis.py` | Merge per-frame CSVs into raw / 1-min / 5-min series |
| `scripts/generate_report.py` | Build the HTML report from the summary CSVs |
| `scripts/export_engine.py` | Export `.pt` → TensorRT `.engine`, once per machine |
| `scripts/count_pipeline.py` | Single-clip debug tool with an overlay video (`--save-video`) |
| `templates/report_template.html` | Report HTML/CSS/JS; the script only fills in numbers |
| `config/cached_feeders.json` | Feeder boxes, near/far group and distance thresholds, computed once per camera |

## Moving between Jetson and x86

1. **TensorRT engines don't transfer between machines.** An engine is compiled for one GPU architecture and TensorRT version. `daily_infer.py` defaults to the portable `.pt`. To use TensorRT, run `export_engine.py` on the target machine.
2. **PyTorch is installed differently.** Jetson needs NVIDIA's ARM wheels. On x86, use the [pytorch.org selector](https://pytorch.org/get-started/locally/) for your CUDA version. For this reason, `requirements.txt` leaves out torch and torchvision.
3. **`tegrastats` exists only on Jetson.** `daily_infer.py` detects it and skips temperature and power sampling when it's missing. The report then leaves out that panel.

## Run

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r pipeline/requirements.txt
pip install torch torchvision --index-url <index for your CUDA version>

# weights: download TNC_v11x_0511.pt from Releases into models/ (see ../models/README.md)

# 1. (optional, recommended) build a TensorRT engine on this machine
python3 pipeline/scripts/export_engine.py --model models/TNC_v11x_0511.pt

# 2. per-frame inference for one day
python3 pipeline/scripts/daily_infer.py \
  --dir /path/to/videos/20250514 \
  --out results/20250514 \
  --model models/TNC_v11x_0511.engine     # or .pt (slower, portable)

# 3. merge + bucket
python3 pipeline/scripts/timeseries_analysis.py \
  --dir results/20250514 --out results/20250514_summary

# 4. report (--tegrastats only on Jetson)
python3 pipeline/scripts/generate_report.py \
  --summary-dir results/20250514_summary \
  --tegrastats results/20250514/tegrastats_batch.csv \
  --title "Feeding rhythm" --date 2025-05-14 \
  --out results/20250514_report.html
```

## Known limitations

- **FP16 vs the thesis setup:** compared with the thesis setup's detections (PyTorch FP32 + TTA), FP16 reaches 0.87 recall and 0.97 precision. P/R/F1 against ground truth has not been re-measured. Trends are reliable; absolute counts may be somewhat low.
- **Don't use INT8 for this pipeline.** It detects about 82 chickens per frame against 229 for FP16, mostly by missing the small birds (see log 3-F).
- **Feeder positions are static.** `cached_feeders.json` is tied to one camera view. If the camera moves, recalibrate. `daily_infer.py` does not detect camera movement.
