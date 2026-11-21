#!/usr/bin/env python3
"""
優化決策矩陣 — 每項優化的「效能 ↑ / 精度 ↓ / 功耗」取捨。

基準 = 論文原始設定（PyTorch FP32 + TTA, conf=0.49, iou=0.60, imgsz=960）。
所有精度指標都以它為 ground truth，回答同一個問題：
    「這項優化快了多少？代價是掉多少偵測？」

同一批影格、同一組參數，逐一配置量測。
"""
import gc, json, re, statistics, subprocess, threading, time
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou

VIDEO = "data/videos/20250501/20250501_0700.mp4"
N_FRAMES = 30
IMGSZ, CONF, IOU, MAX_DET = 960, 0.49, 0.60, 1000
MATCH_IOU = 0.5


class Power(threading.Thread):
    def __init__(s):
        super().__init__(daemon=True); s.w=[]; s.g=[]; s._s=threading.Event(); s.p=None
    def run(s):
        s.p=subprocess.Popen(["tegrastats","--interval","200"],stdout=subprocess.PIPE,text=True)
        for l in s.p.stdout:
            if s._s.is_set(): break
            if m:=re.search(r"VDD_IN (\d+)mW",l): s.w.append(int(m.group(1)))
            if m:=re.search(r"GR3D_FREQ (\d+)%",l): s.g.append(int(m.group(1)))
    def stop(s):
        s._s.set()
        if s.p: s.p.terminate()


def load_frames():
    cap = cv2.VideoCapture(VIDEO); fs=[]
    while len(fs) < N_FRAMES:
        ok, f = cap.read()
        if not ok: break
        fs.append(f)
    cap.release(); return fs


def run_ultralytics(model_path, frames, augment):
    from ultralytics import YOLO
    m = YOLO(model_path)
    for _ in range(3):
        m.predict(frames[0], imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
                  augment=augment, device=0, verbose=False)
    pw = Power(); pw.start(); time.sleep(1)
    boxes, lat = [], []
    for f in frames:
        t=time.perf_counter()
        r=m.predict(f, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
                    augment=augment, device=0, verbose=False)[0]
        lat.append((time.perf_counter()-t)*1000)
        boxes.append(r.boxes.xyxy.cpu().float())
    pw.stop(); time.sleep(.3)
    del m; gc.collect(); torch.cuda.empty_cache()
    return boxes, lat, pw


def run_native(engine, frames):
    from trt_runner import TRTRunner
    r = TRTRunner(engine, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET)
    for _ in range(3): r.infer(frames[0])
    pw = Power(); pw.start(); time.sleep(1)
    boxes, lat = [], []
    for f in frames:
        t=time.perf_counter()
        b,_ = r.infer(f)
        lat.append((time.perf_counter()-t)*1000)
        boxes.append(b.cpu().float())
    pw.stop(); time.sleep(.3)
    del r; gc.collect(); torch.cuda.empty_cache()
    return boxes, lat, pw


def accuracy(pred, ref):
    """以 ref 為 ground truth，算 pred 的 recall / precision / F1。"""
    rs, ps = [], []
    for p, g in zip(pred, ref):
        if len(g)==0: continue
        if len(p)==0: rs.append(0.0); ps.append(0.0); continue
        m = box_iou(p, g)
        ps.append((m.max(dim=1).values >= MATCH_IOU).float().mean().item())
        rs.append((m.max(dim=0).values >= MATCH_IOU).float().mean().item())
    R, P = statistics.mean(rs), statistics.mean(ps)
    return R, P, (2*P*R/(P+R) if P+R else 0.0)


CONFIGS = [
    ("A", "論文原設定 PyTorch FP32 + TTA", "ultra", "models/TNC_v11x_0511.pt",     True),
    ("B", "關閉 TTA",                      "ultra", "models/TNC_v11x_0511.pt",     False),
    ("C", "TensorRT FP16",                 "ultra", "models/TNC_v11x_0511.engine", False),
    ("D", "TensorRT FP16 + 原生推論器",     "native","models/TNC_v11x_0511.engine", None),
    ("E", "TensorRT INT8",                 "native","models/TNC_int8.engine",      None),
    ("F", "TensorRT INT8 混合精度",         "native","models/TNC_int8_mixed.engine",None),
]

if __name__ == "__main__":
    frames = load_frames()
    print(f"影格 {len(frames)} 張，來源 {Path(VIDEO).name}")
    print(f"參數 imgsz={IMGSZ} conf={CONF} iou={IOU} max_det={MAX_DET}")
    print(f"精度基準 = 配置 A（論文原設定），IoU≥{MATCH_IOU} 視為同一目標\n")

    ref, rows = None, []
    for cid, name, kind, path, aug in CONFIGS:
        if not Path(path).exists():
            print(f"[{cid}] 跳過（{path} 不存在）"); continue
        print(f"[{cid}] {name} ...", flush=True)
        boxes, lat, pw = (run_ultralytics(path, frames, aug) if kind=="ultra"
                          else run_native(path, frames))
        if ref is None:
            ref = boxes; R=P=F1=1.0
        else:
            R,P,F1 = accuracy(boxes, ref)
        ms = statistics.mean(lat)
        rows.append({
            "id": cid, "name": name,
            "fps": round(1000/ms, 2), "lat_ms": round(ms, 1),
            "power_w": round(statistics.mean(pw.w)/1000, 2) if pw.w else None,
            "gpu_pct": round(statistics.mean(pw.g), 0) if pw.g else None,
            "det_mean": round(statistics.mean(len(b) for b in boxes), 1),
            "recall": round(R, 4), "precision": round(P, 4), "f1": round(F1, 4),
        })
        print(f"      {rows[-1]['fps']:.2f} FPS | {ms:.1f} ms | "
              f"偵測 {rows[-1]['det_mean']:.0f} | R {R*100:.1f}% P {P*100:.1f}% F1 {F1:.3f}")

    base = rows[0]
    for r in rows:
        r["speedup"] = round(r["fps"]/base["fps"], 2)
        r["fps_per_w"] = round(r["fps"]/r["power_w"], 3) if r["power_w"] else None
        r["det_ratio"] = round(r["det_mean"]/base["det_mean"], 3)

    Path("optimization_matrix.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"\n{'='*104}\n  優化決策矩陣（基準 = A 論文原設定）\n{'='*104}")
    print(f"  {'':<3}{'配置':<26}{'FPS':<8}{'加速':<8}{'延遲ms':<9}"
          f"{'偵測數':<8}{'Recall':<9}{'Prec':<9}{'F1':<8}{'功耗W':<8}{'FPS/W'}")
    for r in rows:
        print(f"  {r['id']:<3}{r['name']:<26}{r['fps']:<8.2f}{r['speedup']:<8.2f}"
              f"{r['lat_ms']:<9.1f}{r['det_mean']:<8.0f}{r['recall']*100:<9.1f}"
              f"{r['precision']*100:<9.1f}{r['f1']:<8.3f}{r['power_w']:<8.2f}{r['fps_per_w']:.2f}")
    print(f"\n→ optimization_matrix.json")
