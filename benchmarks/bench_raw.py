#!/usr/bin/env python3
"""
原生 TensorRT 多串流 benchmark — 三種 engine 走同一條路徑。

為什麼不用 stream_bench.py：
  - trtexec 與 TensorRT API 產出的 engine 沒有 ultralytics 的 metadata 標頭，
    YOLO() 載不了（DLA engine、INT8 engine 都是這種）
  - 統一走 TRTRunner 才是公平比較，順帶拿掉 ultralytics 的 Python 包裝開銷

用法：
  python3 bench_raw.py --engine TNC_v11x_0511.engine --streams 1 2
  python3 bench_raw.py --engine TNC_int8.engine --streams 1 2 3 4
  python3 bench_raw.py --engine TNC_dla0.engine --streams 1 --frames 20
"""
import argparse, json, re, statistics, subprocess, threading, time
from pathlib import Path

import cv2
from trt_runner import TRTRunner

VIDEO_DIR = Path("data/videos/20250501")
REALTIME_FPS = 5.0


class Tegra(threading.Thread):
    def __init__(self, ms=200):
        super().__init__(daemon=True)
        self.ms, self.w, self.gpu, self.tj = ms, [], [], []
        self._s = threading.Event(); self.p = None

    def run(self):
        self.p = subprocess.Popen(["tegrastats", "--interval", str(self.ms)],
                                  stdout=subprocess.PIPE, text=True)
        for line in self.p.stdout:
            if self._s.is_set(): break
            if m := re.search(r"VDD_IN (\d+)mW", line): self.w.append(int(m.group(1)))
            if m := re.search(r"GR3D_FREQ (\d+)%", line): self.gpu.append(int(m.group(1)))
            if m := re.search(r"tj@([\d.]+)C", line): self.tj.append(float(m.group(1)))

    def stop(self):
        self._s.set()
        if self.p: self.p.terminate()

    def summ(self):
        f = lambda x: statistics.mean(x) if x else None
        return {"power_w": f(self.w) and f(self.w)/1000,
                "power_peak_w": max(self.w)/1000 if self.w else None,
                "gpu_util_pct": f(self.gpu), "temp_peak_c": max(self.tj) if self.tj else None}


class Worker(threading.Thread):
    def __init__(self, sid, video, engine, frames, conf, iou, max_det, imgsz):
        super().__init__()
        self.sid, self.video, self.engine, self.frames = sid, video, engine, frames
        self.conf, self.iou, self.max_det, self.imgsz = conf, iou, max_det, imgsz
        self.lat, self.det = [], []
        self.t0 = self.t1 = None; self.err = None

    def run(self):
        try:
            r = TRTRunner(self.engine, imgsz=self.imgsz, conf=self.conf,
                          iou=self.iou, max_det=self.max_det)
            cap = cv2.VideoCapture(str(self.video))
            ok, fr = cap.read()
            for _ in range(3):
                if ok: r.infer(fr)
            self.t0 = time.time(); n = 0
            while n < self.frames:
                ok, fr = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                t = time.perf_counter()
                boxes, _ = r.infer(fr)
                self.lat.append((time.perf_counter()-t)*1000)
                self.det.append(len(boxes)); n += 1
            self.t1 = time.time(); cap.release()
        except Exception as e:
            import traceback; self.err = f"{e}\n{traceback.format_exc()[-400:]}"

    def stats(self):
        if self.err: return {"stream": self.sid, "error": self.err}
        L = sorted(self.lat); w = self.t1 - self.t0
        return {"stream": self.sid, "frames": len(L), "fps": round(len(L)/w, 2),
                "lat_mean_ms": round(statistics.mean(L),1),
                "lat_p95_ms": round(L[int(len(L)*.95)],1),
                "lat_p99_ms": round(L[int(len(L)*.99)],1),
                "det_mean": round(statistics.mean(self.det),1)}


def run_cfg(n, videos, engine, frames, a):
    print(f"\n{'='*66}\n  {n} 路 | {Path(engine).name} | 每路 {frames} 幀\n{'='*66}")
    ws = [Worker(i, videos[i % len(videos)], engine, frames,
                 a.conf, a.iou, a.max_det, a.imgsz) for i in range(n)]
    tg = Tegra(); tg.start(); time.sleep(1.5)
    t0 = time.time()
    for w in ws: w.start()
    for w in ws: w.join()
    wall = time.time()-t0
    tg.stop(); time.sleep(.3); hw = tg.summ()

    per = [w.stats() for w in ws]
    if any("error" in s for s in per):
        for s in per:
            if "error" in s: print(f"  ✗ 串流{s['stream']}: {s['error'][:300]}")
        return None

    agg = sum(s["fps"] for s in per); mn = min(s["fps"] for s in per)
    print(f"  {'路':<4}{'FPS':<9}{'延遲mean':<11}{'p99':<9}{'偵測數':<8}")
    for s in per:
        print(f"  {s['stream']:<4}{s['fps']:<9.2f}{s['lat_mean_ms']:<11.1f}"
              f"{s['lat_p99_ms']:<9.1f}{s['det_mean']:<8.0f}")
    print(f"\n  聚合吞吐 : {agg:.2f} FPS")
    print(f"  最慢一路 : {mn:.2f} FPS   即時餘裕 {mn/REALTIME_FPS:.2f}× "
          f"{'✅' if mn>=REALTIME_FPS else '❌'}")
    if hw["power_w"]:
        print(f"  功耗     : {hw['power_w']:.2f} W (峰值 {hw['power_peak_w']:.2f})"
              f"  → {agg/hw['power_w']:.2f} FPS/W")
    print(f"  GPU      : {hw['gpu_util_pct']:.0f}%   溫度峰值 {hw['temp_peak_c']:.1f}°C")
    return {"engine": Path(engine).name, "n_streams": n, "agg_fps": round(agg,2),
            "min_stream_fps": round(mn,2), "realtime_ok": mn>=REALTIME_FPS,
            "fps_per_watt": round(agg/hw["power_w"],3) if hw["power_w"] else None,
            "per_stream": per, **hw}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--streams", type=int, nargs="+", default=[1,2])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.49)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    res = [r for n in a.streams if (r := run_cfg(n, videos, a.engine, a.frames, a))]
    if len(res) > 1:
        print(f"\n{'='*66}\n  總結\n{'='*66}")
        print(f"  {'路數':<7}{'聚合FPS':<11}{'最慢路':<10}{'功耗W':<9}{'FPS/W':<9}{'GPU%':<7}即時")
        for r in res:
            print(f"  {r['n_streams']:<7}{r['agg_fps']:<11.2f}{r['min_stream_fps']:<10.2f}"
                  f"{r['power_w']:<9.2f}{r['fps_per_watt']:<9.2f}{r['gpu_util_pct']:<7.0f}"
                  f"{'✅' if r['realtime_ok'] else '❌'}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n→ {a.out}")
