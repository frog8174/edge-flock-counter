#!/usr/bin/env python3
"""
Measures board power draw (mW, from the onboard INA3221 sensor via tegrastats)
while continuously running TensorRT inference, vs. idle baseline.

Usage: python3 power_check.py [--seconds 10]
"""
import argparse
import re
import statistics
import subprocess
import threading
import time

import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def sample_power(duration_s, out_list, interval_ms=200):
    proc = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                             stdout=subprocess.PIPE, text=True)
    t_end = time.time() + duration_s
    while time.time() < t_end:
        line = proc.stdout.readline()
        m = re.search(r"VDD_IN (\d+)mW", line) if line else None
        if m:
            out_list.append(int(m.group(1)))
    proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--model", default="models/yolo11n.engine")
    args = ap.parse_args()

    # --- idle baseline ---
    idle = []
    print(f"Sampling idle power for {args.seconds}s (no inference running)...")
    sample_power(args.seconds, idle)
    idle_avg = statistics.mean(idle)
    print(f"  idle avg VDD_IN: {idle_avg:.0f} mW\n")

    # --- under load ---
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    model = YOLO(args.model)
    for _ in range(10):
        model.predict(img, verbose=False)  # warmup

    load = []
    t = threading.Thread(target=sample_power, args=(args.seconds, load))
    t.start()
    print(f"Running continuous inference for {args.seconds}s...")
    n = 0
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        model.predict(img, verbose=False)
        n += 1
    t.join()
    load_avg = statistics.mean(load)

    print(f"  {n} inferences in {args.seconds}s -> {n/args.seconds:.1f} FPS")
    print(f"  under-load avg VDD_IN: {load_avg:.0f} mW\n")

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Idle power:          {idle_avg/1000:.2f} W")
    print(f"Under-load power:    {load_avg/1000:.2f} W")
    print(f"Inference power delta: {(load_avg-idle_avg)/1000:.2f} W")
    print(f"Throughput:          {n/args.seconds:.1f} FPS @ {load_avg/1000:.2f} W total board power")


if __name__ == "__main__":
    main()
