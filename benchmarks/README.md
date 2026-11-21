# Benchmarks

Every number in the README and [`docs/optimization-log.md`](../docs/optimization-log.md) comes from these scripts, run on a Jetson Orin NX with `nvpmodel -m 0 && jetson_clocks`. Run them from the repo root, with videos in `data/videos/20250501/` and models in `models/`.

| Script | Measures | Log section |
|---|---|---|
| `benchmark.py`, `power_check.py` | YOLO11n PyTorch vs TensorRT, idle vs load power (methodology check) | Phase 0 |
| `stream_bench.py` | N concurrent streams: aggregate FPS, per-stream latency, GPU %, power | Phase 1–2 |
| `trt_runner.py` | Native TensorRT inference + GPU NMS, no Ultralytics wrapper | 3-C |
| `bench_raw.py` | Multi-stream benchmark for raw engines (INT8 / DLA) through `trt_runner` | 3-C – 3-G |
| `int8/build_int8.py` | INT8 PTQ engine, 210-frame entropy calibration | 3-D |
| `int8/build_int8_mixed.py` | INT8 with attention + head layers pinned to FP16 | 3-E |
| `dla/gpu_dla_combo.py` | GPU and DLA0 alone and concurrently | 3-B, 3-G |
| `optimization_matrix.py` | Unified sweep of configs A–F, with agreement vs baseline detections | README table |
| `field_scaling.py` | Farm-scale projection: devices, data volume, power | Field scaling |
| `make_visuals.py` | FP32 / FP16 / INT8 side-by-side image | README |

`results/` holds the raw JSON outputs and build logs. Engines are not included: build them on your device (see `../models/README.md`).
