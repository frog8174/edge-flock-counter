#!/usr/bin/env python3
"""
把雞隻偵測模型（TNC_v11x_0511.pt, YOLO11x @960）匯出成 TensorRT FP16 engine。

TensorRT engine 是「編譯到特定 GPU 架構 + 特定 TensorRT 版本」的產物，**不可跨機器搬用**。
每台要跑推論的機器（不管是 Jetson 還是資料中心 GPU 伺服器）都要在自己身上重新跑這支腳本一次。
在 Jetson Orin NX 上，YOLO11x（56.9M 參數）建置一次約需 15-20 分鐘；資料中心 GPU 通常更快。

用法：
  python3 export_engine.py --model models/TNC_v11x_0511.pt
  python3 export_engine.py --model models/TNC_v11x_0511.pt --imgsz 960 --device 0
"""
import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="來源 .pt 權重路徑")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=0)
    ap.add_argument("--half", action="store_true", default=True, help="FP16（預設開）")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"找不到 {model_path} —— 模型權重需另外從原始來源或搭配影片一起搬過來，見 README")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 載入 {model_path} ...")
    m = YOLO(str(model_path))

    print(f"[{time.strftime('%H:%M:%S')}] 匯出 TensorRT{'FP16' if args.half else 'FP32'} engine "
         f"(imgsz={args.imgsz}, batch=1, device={args.device}) ...")
    m.export(format="engine", half=args.half, imgsz=args.imgsz, batch=1, device=args.device)

    print(f"[{time.strftime('%H:%M:%S')}] 完成，耗時 {time.time()-t0:.0f}s")
    print(f"engine 檔會產生在 {model_path.with_suffix('.engine')}")


if __name__ == "__main__":
    main()
