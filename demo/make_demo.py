"""Build the README demo GIFs.

Inputs (from pipeline/scripts/count_pipeline.py --save-video):
  --video       annotated 1080p @ 5 fps clip
  --csv         per-frame log of that clip: frame, video_time_s, chickens_total,
                chickens_around_feeder, around_rate, latency_ms  (counts shown in the HUD)
  --timing-csv  per-frame log recorded on the Jetson Orin NX; only latency_ms is used
                (defaults to --csv when the clip itself was rendered on the Jetson)

Outputs:
  hero.gif          detections + feeder zones + live count/latency HUD
  before_after.gif  real-time replay: thesis baseline vs optimized pipeline

The before/after replay is a *timing* replay: both panels show frames from the
same on-device recording; what differs is which frame each pipeline has
finished by wall-clock time t, computed from latencies measured on the Jetson
(baseline = PyTorch FP32 + TTA, 349.1 ms/frame mean; optimized = per-frame
TensorRT FP16 latencies from the CSV). The camera produces a frame every 200 ms.
"""
import argparse, csv, subprocess
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

CAM_DT = 0.2            # camera period (5 fps)
BASELINE_MS = 349.1     # docs/optimization-log.md, Phase 1, PyTorch FP32 + TTA (2.82 FPS measured)
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, GREEN, RED, AMBER = (255,)*3, (170,)*3, (90, 220, 90), (80, 80, 235), (60, 180, 255)
BG = (24, 24, 24)


def load(video, csv_path, timing_csv):
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f[:100, :620] = (f[:100, :620] * 0.15).astype(np.uint8)  # hide the recorder's own HUD; we redraw it
        frames.append(f)
    rows = list(csv.DictReader(open(csv_path)))
    lat = [float(r["latency_ms"]) for r in csv.DictReader(open(timing_csv))]
    n = min(len(frames), len(rows), len(lat))
    assert n > 0, "no frames / rows"
    for r, l in zip(rows, lat):
        r["latency_ms"] = l  # HUD/replay timing always comes from the Jetson log
    return frames[:n], rows[:n]


def text(img, s, org, scale=0.6, color=WHITE, th=1, bg=False):
    if bg:  # dark plate so labels stay readable on bright frames
        (w, h), b = cv2.getTextSize(s, FONT, scale, th)
        x, y = org
        img[max(y - h - 6, 0):y + b + 4, max(x - 6, 0):x + w + 6] = BG
    cv2.putText(img, s, org, FONT, scale, color, th, cv2.LINE_AA)


def to_gif(frames, out, fps, width):
    tmp = Path(out).with_suffix(".mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-vf", vf, "-loop", "0", str(out)], check=True)
    print(f"[ok] {out}  ({Path(out).stat().st_size/1e6:.1f} MB)  + {tmp}")


def hero(frames, rows, seconds):
    W, H, PH = 960, 540, 130
    around = np.array([int(r["chickens_around_feeder"]) for r in rows])
    lat = np.array([float(r["latency_ms"]) for r in rows])
    n = min(len(frames), int(seconds / CAM_DT))
    out = []
    for i in range(n):
        canvas = np.full((H + PH, W, 3), BG, np.uint8)
        canvas[:H] = cv2.resize(frames[i], (W, H), interpolation=cv2.INTER_AREA)
        r = rows[i]
        text(canvas, f"Around feeders: {r['chickens_around_feeder']} / {r['chickens_total']} chickens", (12, 28), 0.7, GREEN, 2, bg=True)
        text(canvas, "white ring = feeder zone (per-row distance threshold)", (12, 54), 0.5, GREY, bg=True)
        y0 = H + 28
        text(canvas, "Jetson Orin NX  |  YOLO11x @960  |  TensorRT FP16", (12, y0), 0.6, WHITE)
        text(canvas, f"latency {lat[i]:5.1f} ms   {1000/lat[i]:4.1f} FPS   camera 5 FPS -> real-time", (12, y0 + 28), 0.6, GREEN)
        text(canvas, "17.8 W board power  |  2 cameras real-time per device", (12, y0 + 56), 0.55, GREY)
        # sparkline: chickens around feeders over the clip, cursor at current frame
        x0, x1, ys, ye = 600, 945, H + 15, H + PH - 15
        seg = around[:n]
        lo, hi = seg.min(), max(seg.max(), seg.min() + 1)
        pts = np.stack([np.linspace(x0, x1, len(seg)), ye - (seg - lo) / (hi - lo) * (ye - ys)], 1).astype(np.int32)
        cv2.polylines(canvas, [pts[: i + 1]], False, GREEN, 2, cv2.LINE_AA)
        cv2.polylines(canvas, [pts[i:]], False, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[i]), 4, WHITE, -1, cv2.LINE_AA)
        text(canvas, "around-feeder count", (x0, ys - 2), 0.4, GREY)
        out.append(canvas)
    return out


def panel_frame(done_idx, frames, size):
    return cv2.resize(frames[done_idx], size, interpolation=cv2.INTER_AREA)


def before_after(frames, rows, seconds):
    PW, PH, HEAD, FOOT = 640, 360, 70, 60
    lat = [float(r["latency_ms"]) / 1000 for r in rows]
    # optimized: frame k is captured at k*CAM_DT, processed when the GPU is free
    done_opt, t = [], 0.0
    for k, l in enumerate(lat):
        t = max(t, k * CAM_DT) + l
        done_opt.append(t)
    out = []
    n_cam = min(len(frames), int(seconds / CAM_DT))
    for k in range(n_cam):
        now = k * CAM_DT + CAM_DT  # render at camera cadence
        captured = min(int(now / CAM_DT), len(frames) - 1)
        # baseline: always works on the oldest queued frame, one per 349 ms
        b_idx = min(int(now / (BASELINE_MS / 1000)) - 1, len(frames) - 1)
        o_idx = max(i for i, d in enumerate(done_opt) if d <= now) if done_opt[0] <= now else -1
        canvas = np.full((HEAD + PH + FOOT, PW * 2 + 10, 3), BG, np.uint8)
        text(canvas, f"camera time  {now:5.1f} s   (5 FPS, frame #{captured})", (12, 26), 0.65, WHITE, 1)
        text(canvas, "same on-device recording; timing replayed from latencies measured on Jetson Orin NX", (12, 52), 0.45, GREY)
        for col, (idx, title, sub, color) in enumerate([
            (b_idx, "BEFORE  PyTorch FP32 + TTA", "2.82 FPS  (thesis setup)", AMBER),
            (o_idx, "AFTER  TensorRT FP16", "11.0 FPS  (optimized)", GREEN),
        ]):
            x = col * (PW + 10)
            if idx >= 0:
                canvas[HEAD:HEAD + PH, x:x + PW] = panel_frame(idx, frames, (PW, PH))
            lag = now - idx * CAM_DT if idx >= 0 else now
            behind = captured - idx if idx >= 0 else captured
            text(canvas, title, (x + 10, HEAD + 26), 0.65, color, 2, bg=True)
            text(canvas, sub, (x + 10, HEAD + 52), 0.5, WHITE, bg=True)
            ok = lag < 0.5
            text(canvas, f"lag {lag:4.1f} s   backlog {behind:3d} frames", (x + 10, HEAD + PH + 26), 0.6, GREEN if ok else RED, 2)
            text(canvas, "keeps up with the camera" if ok else "falling behind, backlog grows forever",
                 (x + 10, HEAD + PH + 50), 0.5, GREEN if ok else RED)
        out.append(canvas)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timing-csv", default=None)
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    frames, rows = load(a.video, a.csv, a.timing_csv or a.csv)
    to_gif(hero(frames, rows, 12), f"{a.out}/hero.gif", 5, 720)
    to_gif(before_after(frames, rows, a.seconds), f"{a.out}/before_after.gif", 5, 900)


if __name__ == "__main__":
    main()
