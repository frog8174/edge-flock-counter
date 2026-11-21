#!/usr/bin/env python3
"""
整天雞舍影片批次推論 → 逐幀時序 CSV。

跟 count_pipeline.py 的差異：
  - 一次跑一整個資料夾（一天份的 mp4），不是單支影片
  - 每支影片只跑一次（不用 --frames 限制），輸出用「絕對時間」而非影片內相對時間
    （檔名 20250514_0700.mp4 → 這支影片第 n 幀 = 07:00:00 + n/5 秒，5 可用 --fps-fallback 調整）
  - 每支影片各自寫一個 CSV，可中斷、可續跑（已完成的影片會跳過）
    → 資料還在陸續到位時可以先跑已到的部分，之後補跑新到的檔案即可
  - 整個批次期間只跑「一次」tegrastats 背景取樣（僅 Jetson 平台，見下方 PORTABILITY NOTE），
    量到的是長時間（數小時）穩態下的 GPU/功耗/溫度趨勢

PORTABILITY NOTE（從 Jetson 移到別的 server 時要注意）：
  - `tegrastats` 是 Jetson 專用工具，其他平台（x86 + 獨立顯卡）不存在，
    本腳本會自動偵測、偵測不到就跳過溫度/功耗取樣，不會噴錯。
  - --model 預設吃 .pt（PyTorch 權重，跨平台通用）。若要用 TensorRT engine 加速，
    engine 檔是「編譯到特定 GPU 架構 + TensorRT 版本」的產物，**不能跨機器搬用**，
    要在目標機器上用 export_engine.py 重新匯出。
  - --device 預設 0（第一張 GPU），多卡機器請自行指定。

用法：
  python3 daily_infer.py --dir /path/to/videos/20250514 --out results/20250514
  python3 daily_infer.py --dir ... --out ... --model models/TNC_v11x_0511.engine  # 用重新匯出的 TensorRT engine
"""
import argparse, csv, json, re, shutil, subprocess, threading, time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEEDER_JSON = SCRIPT_DIR.parent / "config" / "cached_feeders.json"
DEFAULT_MODEL = SCRIPT_DIR.parent.parent / "models" / "TNC_v11x_0511.pt"
THRESH = [45.10, 94.23]        # group0(遠) / group1(近) 距離門檻 (px) — 綁定原相機視角，換相機需重算
FNAME_RE = re.compile(r"(\d{8})_(\d{4})\.mp4$")


class TegraSampler(threading.Thread):
    """整個批次跑一次，不是每支影片跑一次 —— 才能看出數小時尺度的穩態趨勢。
    僅 Jetson 平台可用；建構前請先用 TegraSampler.available() 檢查。"""
    def __init__(self, out_path, interval_ms=2000):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.out_path = out_path
        self.proc = None
        self._stop = threading.Event()

    @staticmethod
    def available():
        return shutil.which("tegrastats") is not None

    def run(self):
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=subprocess.PIPE, text=True)
        with open(self.out_path, "a") as fh:
            for line in self.proc.stdout:
                if self._stop.is_set():
                    break
                w = re.search(r"VDD_IN (\d+)mW", line)
                gpu = re.search(r"GR3D_FREQ (\d+)%", line)
                temp = re.search(r"tj@(\d+\.?\d*)C", line)
                fh.write(f"{time.time():.1f},{w.group(1) if w else ''},"
                        f"{gpu.group(1) if gpu else ''},{temp.group(1) if temp else ''}\n")
                fh.flush()

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()


def load_feeders(feeder_json):
    feeders = json.load(open(feeder_json))
    centers = np.array([[f["bbox"][0] + f["bbox"][2] / 2,
                         f["bbox"][1] + f["bbox"][3] / 2] for f in feeders])
    groups = np.array([f["group"] for f in feeders])
    radii = np.array([THRESH[g] for g in groups])
    return centers, groups, radii


def parse_start_time(fname):
    m = FNAME_RE.search(fname)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def process_video(model, path, f_cent, f_group, f_rad, imgsz, conf, iou, max_det,
                  device, fps_fallback, out_csv):
    start_t = parse_start_time(path.name)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ⚠️ 無法開啟 {path.name}，跳過")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback

    rows = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                          augment=False, device=device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.empty((0, 4))
        total = len(xyxy)
        around = g0 = g1 = 0
        if total:
            cxy = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2,
                            (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)
            d = np.linalg.norm(cxy[:, None, :] - f_cent[None, :, :], axis=2)
            nearest = d.argmin(axis=1)
            flags = d[np.arange(total), nearest] <= f_rad[nearest]
            around = int(flags.sum())
            grp_of_flagged = f_group[nearest[flags]]
            g0 = int((grp_of_flagged == 0).sum())
            g1 = int((grp_of_flagged == 1).sum())

        ts = start_t + timedelta(seconds=n / src_fps) if start_t else None
        rows.append({
            "timestamp": ts.isoformat() if ts else "",
            "video_file": path.name,
            "frame": n,
            "chickens_total": total,
            "chickens_around_feeder": around,
            "around_group0_far": g0,
            "around_group1_near": g1,
            "around_rate": round(around / total, 4) if total else 0.0,
        })
        n += 1
    cap.release()

    with open(out_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="一天份影片所在資料夾")
    ap.add_argument("--out", required=True, help="每支影片各自的逐幀 CSV 輸出資料夾")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="模型路徑：.pt（跨平台，預設）或本機重新匯出的 .engine（較快，不可跨機器搬用）")
    ap.add_argument("--feeders", default=str(DEFAULT_FEEDER_JSON), help="餵食器座標 + 分組 JSON")
    ap.add_argument("--device", default=0, help="推論裝置，例如 0 或 'cpu'")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.49)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--fps-fallback", type=float, default=5.0,
                    help="影片檔案本身沒有 fps metadata 時的備用值（現場相機規格）")
    args = ap.parse_args()

    src_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(src_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"{src_dir} 底下沒有找到 .mp4 影片")
    print(f"[{datetime.now():%H:%M:%S}] 找到 {len(videos)} 支影片於 {src_dir}")

    f_cent, f_group, f_rad = load_feeders(args.feeders)
    print(f"餵食器：{len(f_cent)} 個 (group0={int((f_group==0).sum())} 遠, "
          f"group1={int((f_group==1).sum())} 近)")

    print(f"[{datetime.now():%H:%M:%S}] 載入模型 {args.model} ...")
    model = YOLO(args.model)
    # 暖機
    warm = cv2.VideoCapture(str(videos[0]))
    ok, frame = warm.read(); warm.release()
    if ok:
        for _ in range(3):
            model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          max_det=args.max_det, augment=False, device=args.device, verbose=False)

    tegra_log = out_dir / "tegrastats_batch.csv"
    sampler = None
    if TegraSampler.available():
        if not tegra_log.exists():
            tegra_log.write_text("unix_ts,power_mw,gpu_pct,temp_c\n")
        sampler = TegraSampler(tegra_log)
        sampler.start()
    else:
        print("ℹ️  找不到 tegrastats（非 Jetson 平台），跳過溫度/功耗取樣")

    t_batch_start = time.time()
    done = skipped = 0
    for i, vpath in enumerate(videos, 1):
        out_csv = out_dir / f"{vpath.stem}.csv"
        if out_csv.exists():
            skipped += 1
            continue
        t0 = time.time()
        n = process_video(model, vpath, f_cent, f_group, f_rad,
                          args.imgsz, args.conf, args.iou, args.max_det,
                          args.device, args.fps_fallback, out_csv)
        dt = time.time() - t0
        done += 1
        elapsed = time.time() - t_batch_start
        print(f"[{datetime.now():%H:%M:%S}] ({i}/{len(videos)}) {vpath.name}: "
              f"{n} 幀, {dt:.1f}s ({n/dt:.2f} FPS)  "
              f"累計進度 {done} 支新完成 / {skipped} 支已跳過  "
              f"累計耗時 {elapsed/60:.1f} 分")

    if sampler:
        sampler.stop(); time.sleep(0.3)
    total_elapsed = time.time() - t_batch_start
    print(f"\n[{datetime.now():%H:%M:%S}] 批次完成：新處理 {done} 支，"
          f"跳過(已存在) {skipped} 支，耗時 {total_elapsed/60:.1f} 分鐘")
    print(f"逐幀 CSV → {out_dir}/*.csv")
    if sampler:
        print(f"批次期間 tegrastats 取樣 → {tegra_log}")


if __name__ == "__main__":
    main()
