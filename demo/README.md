# Demo assets

Rebuild the README GIFs (both play at 5 FPS, the camera rate, i.e. real time) and chart from on-device outputs (`pip install opencv-python-headless imageio-ffmpeg matplotlib pandas`):

```bash
# render.mp4 / render.csv: count_pipeline.py --save-video (any GPU; used for boxes and counts)
# inference_result.csv: the same clip run on the Jetson (used only for latency)
python3 pipeline/scripts/count_pipeline.py --video data/videos/20250501/20250501_0700.mp4 \
  --model models/TNC_v11x_0511.pt --frames 150 --no-augment --save-video render.mp4 --csv render.csv
python3 demo/make_demo.py --video render.mp4 --csv render.csv --timing-csv inference_result.csv --out docs/assets
python3 demo/plot_rhythm.py docs/assets/feeding_rhythm_20250514_5min.csv docs/assets/feeding_rhythm.png
```

`before_after.gif` is a **timing replay**: both panels use frames from the same on-device recording. Which frame each pipeline has finished at time *t* comes from latencies measured on the Jetson: 349.1 ms/frame for the thesis setup, and the per-frame TensorRT FP16 latencies from the CSV.
