#!/usr/bin/env python3
"""
雞舍多串流邊緣推論 Benchmark — 單路 vs 雙路 vs N 路

回答的問題：一台 Jetson Orin NX 能同時即時處理幾路雞舍攝影機？

現場需求（來自簡報）：
    - 每台 Raspberry Pi 4B + 130° 相機，1920×1080 @ **5 fps**
    - 雞舍 113m × 14.8m × 10 排，Future Work 要「涵蓋更大範圍」
    - 每幀 650~750 隻雞，模型 YOLO11x @ 960

因此「即時」的門檻是 **5 fps/路**，不是 30 fps。這是本專案的關鍵 edge 特性。

量測項目：
    - 每路 FPS / 聚合 FPS
    - 單幀延遲 (mean / p50 / p95 / p99)  ← edge 應用在意尾延遲
    - 即時餘裕 (achieved FPS ÷ 5 fps 需求)
    - 板級功耗 (VDD_IN)、FPS/W 能效
    - GPU 頻率、溫度

用法：
    python3 stream_bench.py --streams 1
    python3 stream_bench.py --streams 2
    python3 stream_bench.py --streams 1 2 4 --frames 200
"""
import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_DIR = Path("data/videos/20250501")
REALTIME_FPS = 5.0          # 現場相機幀率 = 即時門檻


# ─────────────────────────── 功耗 / 溫度取樣 ───────────────────────────
class TegraSampler(threading.Thread):
    """背景跑 tegrastats，收集功耗、GPU 使用率、溫度。"""

    def __init__(self, interval_ms=200):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.vdd_in, self.gr3d, self.tj = [], [], []
        self._stop = threading.Event()
        self.proc = None

    def run(self):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, text=True,
        )
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            if m := re.search(r"VDD_IN (\d+)mW", line):
                self.vdd_in.append(int(m.group(1)))
            if m := re.search(r"GR3D_FREQ (\d+)%", line):
                self.gr3d.append(int(m.group(1)))
            if m := re.search(r"tj@([\d.]+)C", line):
                self.tj.append(float(m.group(1)))

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()

    def summary(self):
        f = lambda xs: statistics.mean(xs) if xs else None
        return {
            "power_w": (f(self.vdd_in) / 1000) if self.vdd_in else None,
            "power_peak_w": (max(self.vdd_in) / 1000) if self.vdd_in else None,
            "gpu_util_pct": f(self.gr3d),
            "gpu_util_peak_pct": max(self.gr3d) if self.gr3d else None,
            "temp_c": f(self.tj),
            "temp_peak_c": max(self.tj) if self.tj else None,
        }


# ─────────────────────────── 單一串流 worker ───────────────────────────
class StreamWorker(threading.Thread):
    """一路串流 = 一個獨立的解碼 + 推論迴圈（各自的 model context）。"""

    def __init__(self, sid, video, model_path, frames, imgsz, conf, iou, max_det, augment):
        super().__init__()
        self.sid, self.video, self.model_path = sid, video, model_path
        self.frames, self.imgsz = frames, imgsz
        self.conf, self.iou, self.max_det, self.augment = conf, iou, max_det, augment
        self.latencies = []      # 每幀端到端延遲 (ms)
        self.det_counts = []     # 每幀偵測到的雞數
        self.t_start = self.t_end = None
        self.error = None

    def run(self):
        try:
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(str(self.video))
            if not cap.isOpened():
                raise RuntimeError(f"無法開啟 {self.video}")

            # 暖機（讓 CUDA context / engine 就緒，不計入統計）
            ok, frame = cap.read()
            if ok:
                for _ in range(3):
                    model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                  max_det=self.max_det, augment=self.augment,
                                  device=0, verbose=False)

            self.t_start = time.time()
            n = 0
            while n < self.frames:
                ok, frame = cap.read()
                if not ok:                      # 影片結束就從頭再來（模擬連續串流）
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                t0 = time.perf_counter()
                r = model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                  max_det=self.max_det, augment=self.augment,
                                  device=0, verbose=False)[0]
                self.latencies.append((time.perf_counter() - t0) * 1000)
                self.det_counts.append(len(r.boxes))
                n += 1
            self.t_end = time.time()
            cap.release()
        except Exception as e:
            self.error = str(e)

    def stats(self):
        if self.error or not self.latencies:
            return {"stream": self.sid, "error": self.error}
        lat = sorted(self.latencies)
        wall = self.t_end - self.t_start
        return {
            "stream": self.sid,
            "video": self.video.name,
            "frames": len(self.latencies),
            "wall_s": round(wall, 2),
            "fps": round(len(self.latencies) / wall, 2),
            "lat_mean_ms": round(statistics.mean(lat), 1),
            "lat_p50_ms": round(lat[len(lat) // 2], 1),
            "lat_p95_ms": round(lat[int(len(lat) * 0.95)], 1),
            "lat_p99_ms": round(lat[int(len(lat) * 0.99)], 1),
            "det_mean": round(statistics.mean(self.det_counts), 1),
            "det_max": max(self.det_counts),
        }


# ─────────────────────────── 一次 N 路測試 ───────────────────────────
def run_config(n_streams, videos, model_path, frames, args):
    print(f"\n{'='*68}")
    print(f"  {n_streams} 路串流  |  每路 {frames} 幀  |  imgsz={args.imgsz}  "
          f"TTA={'ON' if args.augment else 'OFF'}")
    print(f"{'='*68}")

    workers = [
        StreamWorker(i, videos[i % len(videos)], model_path, frames,
                     args.imgsz, args.conf, args.iou, args.max_det, args.augment)
        for i in range(n_streams)
    ]

    sampler = TegraSampler()
    sampler.start()
    time.sleep(1.5)                      # 讓 sampler 先抓到基線

    t0 = time.time()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    wall = time.time() - t0

    sampler.stop()
    time.sleep(0.3)
    hw = sampler.summary()

    per = [w.stats() for w in workers]
    if any("error" in s for s in per):
        for s in per:
            if "error" in s:
                print(f"  ✗ stream {s['stream']}: {s['error']}")
        return None

    agg_frames = sum(s["frames"] for s in per)
    agg_fps_wall = agg_frames / wall          # 含模型載入，偏悲觀
    agg_fps = sum(s["fps"] for s in per)      # 各路穩態加總，反映真實吞吐
    per_fps = [s["fps"] for s in per]
    min_fps = min(per_fps)

    print(f"\n  每路明細：")
    print(f"  {'路':<4}{'FPS':<9}{'延遲mean':<11}{'p95':<9}{'p99':<9}{'偵測數':<8}")
    for s in per:
        print(f"  {s['stream']:<4}{s['fps']:<9.2f}{s['lat_mean_ms']:<11.1f}"
              f"{s['lat_p95_ms']:<9.1f}{s['lat_p99_ms']:<9.1f}{s['det_mean']:<8.0f}")

    margin = min_fps / REALTIME_FPS
    realtime = "✅ 達標" if min_fps >= REALTIME_FPS else "❌ 掉幀"

    print(f"\n  聚合吞吐 : {agg_fps:.2f} FPS (穩態加總)  |  {agg_fps_wall:.2f} FPS (含載入)")
    print(f"  最慢一路 : {min_fps:.2f} FPS")
    print(f"  即時餘裕 : {margin:.2f}× (需求 {REALTIME_FPS} fps/路)  {realtime}")
    if hw["power_w"]:
        print(f"  功耗     : 平均 {hw['power_w']:.2f} W / 峰值 {hw['power_peak_w']:.2f} W")
        print(f"  能效     : {agg_fps/hw['power_w']:.2f} FPS/W")
    if hw["gpu_util_pct"] is not None:
        print(f"  GPU 使用率: 平均 {hw['gpu_util_pct']:.0f}% / 峰值 {hw['gpu_util_peak_pct']}%")
    if hw["temp_c"]:
        print(f"  溫度     : 平均 {hw['temp_c']:.1f}°C / 峰值 {hw['temp_peak_c']:.1f}°C")

    return {
        "n_streams": n_streams, "frames_per_stream": frames,
        "imgsz": args.imgsz, "augment": args.augment,
        "agg_fps": round(agg_fps, 2), "agg_fps_wall": round(agg_fps_wall, 2), "min_stream_fps": round(min_fps, 2),
        "realtime_margin": round(margin, 2),
        "realtime_ok": min_fps >= REALTIME_FPS,
        "wall_s": round(wall, 2),
        "fps_per_watt": round(agg_fps / hw["power_w"], 3) if hw["power_w"] else None,
        "per_stream": per, **hw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/TNC_v11x_0511.engine",
                    help="模型路徑 (.engine 或 .pt)")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2],
                    help="要測試的串流數，可給多個")
    ap.add_argument("--frames", type=int, default=100, help="每路測幾幀")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.49)   # 對齊原專案設定
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--augment", action="store_true", help="開啟 TTA（原專案有開，成本約3倍）")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"在 {VIDEO_DIR} 找不到影片")
    print(f"影片來源：{VIDEO_DIR}（{len(videos)} 支，1920×1080 @ 5fps）")
    print(f"模型：{args.model}")

    results = []
    for n in args.streams:
        r = run_config(n, videos, args.model, args.frames, args)
        if r:
            results.append(r)

    # ── 總結表 ──
    if len(results) > 1:
        print(f"\n{'='*68}")
        print("  總結：串流數擴展性")
        print(f"{'='*68}")
        print(f"  {'路數':<7}{'聚合FPS':<11}{'最慢路FPS':<12}{'餘裕':<9}"
              f"{'功耗W':<9}{'FPS/W':<9}{'GPU%':<7}{'即時'}")
        for r in results:
            print(f"  {r['n_streams']:<7}{r['agg_fps']:<11.2f}{r['min_stream_fps']:<12.2f}"
                  f"{r['realtime_margin']:<9.2f}{r['power_w'] or 0:<9.2f}"
                  f"{r['fps_per_watt'] or 0:<9.2f}{r['gpu_util_pct'] or 0:<7.0f}"
                  f"{'✅' if r['realtime_ok'] else '❌'}")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n結果已存到 {args.out}")


if __name__ == "__main__":
    main()
