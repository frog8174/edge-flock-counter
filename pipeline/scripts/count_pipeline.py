#!/usr/bin/env python3
"""
雞舍餵食器計數 — Jetson 邊緣即時版

對照原專案 (EatingChickenDetect.py) 的三處移植修正：
  1. device=3 → device=0        （原本寫死實驗室多卡機的第 4 張卡）
  2. NAS 絕對路徑 → 本機路徑
  3. 餵食器偵測模型 → 直接讀 cached_feeders.json
     相機固定不動，餵食器位置是靜態的，本來就只需算一次。
     省掉一整個模型的每幀推論 —— 這是免費的算力。

輸出：
  - 每幀「餵食器周圍雞數 / 總雞數」時序 CSV
  - 可選的標註影片
  - 效能與功耗統計

用法：
  python3 count_pipeline.py --video data/videos/20250501/20250501_0700.mp4 --frames 100
  python3 count_pipeline.py --video ... --save-video out.mp4 --no-augment
"""
import argparse, csv, json, re, shutil, statistics, subprocess, threading, time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# label the overlay with the real device instead of assuming Jetson
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

SCRIPT_DIR = Path(__file__).resolve().parent
FEEDER_JSON = SCRIPT_DIR.parent / "config" / "cached_feeders.json"
THRESH = [45.10, 94.23]        # group0(遠/小) / group1(近/大) 的距離門檻 (px)


class PowerSampler(threading.Thread):
    """背景取樣 tegrastats 的板級功耗與 GPU 使用率。僅 Jetson 平台可用（見 available()）。"""
    @staticmethod
    def available():
        return shutil.which("tegrastats") is not None

    def __init__(self, interval_ms=200):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.w, self.gpu = [], []
        self.proc = None
        self._stop = threading.Event()

    def run(self):
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=subprocess.PIPE, text=True)
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            if m := re.search(r"VDD_IN (\d+)mW", line):
                self.w.append(int(m.group(1)))
            if m := re.search(r"GR3D_FREQ (\d+)%", line):
                self.gpu.append(int(m.group(1)))

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()


def load_feeders():
    feeders = json.load(open(FEEDER_JSON))
    centers = np.array([[f["bbox"][0] + f["bbox"][2] / 2,
                         f["bbox"][1] + f["bbox"][3] / 2] for f in feeders])
    groups = np.array([f["group"] for f in feeders])
    radii = np.array([THRESH[g] for g in groups])
    return feeders, centers, groups, radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=str(SCRIPT_DIR.parent.parent / "models" / "TNC_v11x_0511.pt"))
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.49)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--augment", dest="augment", action="store_true",
                    help="開啟 TTA（原專案預設有開，成本約 3 倍）")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.set_defaults(augment=False)
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--csv", default="feeding_timeseries.csv")
    args = ap.parse_args()

    feeders, f_cent, f_group, f_rad = load_feeders()
    print(f"餵食器：{len(feeders)} 個 "
          f"(group0={int((f_group==0).sum())} 遠, group1={int((f_group==1).sum())} 近)")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟 {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    print(f"影片：{Path(args.video).name}  來源 {src_fps:g} fps  "
          f"TTA={'ON' if args.augment else 'OFF'}  imgsz={args.imgsz}")

    writer = None
    rows, lat = [], []

    # 暖機
    ok, frame = cap.read()
    if ok:
        for _ in range(3):
            model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          max_det=args.max_det, augment=args.augment,
                          device=0, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ps = PowerSampler() if PowerSampler.available() else None
    if ps: ps.start()
    time.sleep(1.0)
    t_start = time.time()
    n = 0
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          max_det=args.max_det, augment=args.augment,
                          device=0, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)

        # ── 判定每隻雞是否在某個餵食器的門檻內 ──
        xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.empty((0, 4))
        total = len(xyxy)
        around = 0
        flags = np.zeros(total, dtype=bool)
        if total:
            cxy = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2,
                            (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)
            d = np.linalg.norm(cxy[:, None, :] - f_cent[None, :, :], axis=2)  # (N, F)
            nearest = d.argmin(axis=1)
            flags = d[np.arange(total), nearest] <= f_rad[nearest]
            around = int(flags.sum())

        ts = n / src_fps
        rows.append({"frame": n, "video_time_s": round(ts, 2),
                     "chickens_total": total, "chickens_around_feeder": around,
                     "around_rate": round(around / total, 4) if total else 0.0,
                     "latency_ms": round(lat[-1], 1)})

        if args.save_video:
            vis = frame.copy()
            for (cx, cy), rad in zip(f_cent, f_rad):
                cv2.circle(vis, (int(cx), int(cy)), int(rad), (255, 255, 255), 2)
            # draw around-feeder boxes last and thicker, so overlapping neighbours can't hide their edges
            for i in np.argsort(flags, kind="stable"):
                box, fl = xyxy[i].astype(int), flags[i]
                c, th = ((0, 255, 0), 2) if fl else ((200, 200, 255), 1)
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), c, th)
            rate = around / total * 100 if total else 0
            cv2.putText(vis, f"Around feeder: {rate:.1f}% ({around}/{total})",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
            cv2.putText(vis, f"Around feeder: {rate:.1f}% ({around}/{total})",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(vis, f"{1000/lat[-1]:.1f} FPS  on {DEVICE_NAME}",
                        (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(vis, f"{1000/lat[-1]:.1f} FPS  on {DEVICE_NAME}",
                        (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(args.save_video,
                                         cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(vis)
        n += 1

    wall = time.time() - t_start
    if ps: ps.stop(); time.sleep(0.3)
    cap.release()
    if writer:
        writer.release()

    with open(args.csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    s = sorted(lat)
    fps = n / wall
    avg_w = statistics.mean(ps.w) / 1000 if ps and ps.w else None
    tot = [r["chickens_total"] for r in rows]
    ar = [r["chickens_around_feeder"] for r in rows]

    print(f"\n{'='*60}\n  結果\n{'='*60}")
    print(f"  處理幀數   : {n}  ({wall:.1f}s)")
    print(f"  吞吐       : {fps:.2f} FPS")
    print(f"  即時餘裕   : {fps/src_fps:.2f}× (現場需求 {src_fps:g} fps)"
          f"  {'✅' if fps >= src_fps else '❌'}")
    print(f"  延遲       : mean {statistics.mean(s):.1f}ms | "
          f"p50 {s[len(s)//2]:.1f} | p95 {s[int(len(s)*.95)]:.1f} | p99 {s[int(len(s)*.99)]:.1f}")
    print(f"  雞隻總數   : 平均 {statistics.mean(tot):.0f} / 幀 (max {max(tot)})")
    print(f"  餵食器周圍 : 平均 {statistics.mean(ar):.1f} 隻 "
          f"({statistics.mean([r['around_rate'] for r in rows])*100:.1f}%)")
    if avg_w:
        print(f"  功耗       : {avg_w:.2f} W  →  能效 {fps/avg_w:.2f} FPS/W")
    if ps and ps.gpu:
        print(f"  GPU 使用率 : 平均 {statistics.mean(ps.gpu):.0f}% / 峰值 {max(ps.gpu)}%")
    print(f"\n  時序資料 → {args.csv}")
    if args.save_video:
        print(f"  標註影片 → {args.save_video}")


if __name__ == "__main__":
    main()
