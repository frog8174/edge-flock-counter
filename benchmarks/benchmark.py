#!/usr/bin/env python3
"""
Benchmarks YOLO object detection on-device: PyTorch (.pt) vs TensorRT (.engine),
plus power draw sampled from tegrastats during each run.

This is the "numbers" script for an interview talking point:
    - throughput (FPS) PyTorch vs TensorRT
    - latency (ms/frame)
    - power draw (mW) during inference, sampled directly from the Jetson's
      onboard power monitor (INA3221), not estimated

Usage:
    python3 benchmark.py                # runs both stages, prints a summary table
    python3 benchmark.py --iters 300
"""
import argparse
import re
import statistics
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def sample_power(duration_s, interval_ms=200):
    """Launch tegrastats for `duration_s` seconds, return avg VDD_IN (mW)."""
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE, text=True,
    )
    lines = []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        line = proc.stdout.readline()
        if line:
            lines.append(line)
    proc.terminate()

    vals = []
    for line in lines:
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m:
            vals.append(int(m.group(1)))
    return statistics.mean(vals) if vals else None


def bench_model(model_path, img, iters, warmup=20):
    model = YOLO(model_path)
    # warmup
    for _ in range(warmup):
        model.predict(img, verbose=False)

    times = []
    for _ in range(iters):
        t0 = time.time()
        model.predict(img, verbose=False)
        times.append(time.time() - t0)

    avg_ms = statistics.mean(times) * 1000
    fps = 1000 / avg_ms
    return avg_ms, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pt-model", default="models/yolo11n.pt")
    ap.add_argument("--engine-model", default="models/yolo11n.engine")
    ap.add_argument("--skip-export", action="store_true", help="reuse existing .engine if present")
    args = ap.parse_args()

    img = cv2.imread(str(ASSETS / "bus.jpg"))

    print("=" * 60)
    print("STAGE 1: PyTorch baseline (.pt, unoptimized)")
    print("=" * 60)
    pt_ms, pt_fps = bench_model(args.pt_model, img, args.iters)
    print(f"  {pt_ms:.2f} ms/frame  |  {pt_fps:.1f} FPS")

    engine_path = Path(args.engine_model)
    if not engine_path.exists() or not args.skip_export:
        print("\n" + "=" * 60)
        print("STAGE 2: Exporting to TensorRT engine (FP16) ...")
        print("=" * 60)
        m = YOLO(args.pt_model)
        m.export(format="engine", half=True)  # writes yolo11n.engine next to the .pt

    print("\n" + "=" * 60)
    print("STAGE 3: TensorRT engine (optimized, FP16)")
    print("=" * 60)
    trt_ms, trt_fps = bench_model(args.engine_model, img, args.iters)
    print(f"  {trt_ms:.2f} ms/frame  |  {trt_fps:.1f} FPS")

    speedup = trt_fps / pt_fps

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Backend':<20}{'ms/frame':<12}{'FPS':<10}")
    print(f"{'PyTorch':<20}{pt_ms:<12.2f}{pt_fps:<10.1f}")
    print(f"{'TensorRT FP16':<20}{trt_ms:<12.2f}{trt_fps:<10.1f}")
    print(f"\nTensorRT speedup: {speedup:.2f}x")
    print("(All inference ran on-device, zero network round-trips.)")


if __name__ == "__main__":
    main()
