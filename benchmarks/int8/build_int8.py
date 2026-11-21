#!/usr/bin/env python3
"""
建 INT8 engine — 用真實雞舍影格做校正。

Orin 的 INT8 算力是 FP16 的兩倍。但 INT8 需要校正：
用實際場景的影像統計每層的動態範圍，才不會掉精度。
校正集刻意跨 0700→1718 七個時段，涵蓋清晨到傍晚的光照變化。
"""
import os
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

ONNX = "models/TNC_v11x_0511.onnx"
OUT = "models/TNC_int8.engine"
CACHE = "int8_calib.cache"
CALIB_DIR = Path("calib")
IMGSZ = 960
BATCH = 1

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def letterbox(img, size=IMGSZ):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, files, cache=CACHE):
        super().__init__()
        self.files = files
        self.cache = cache
        self.idx = 0
        import torch
        self.torch = torch
        # 用 torch 配置裝置記憶體，省掉 pycuda 相依（這台沒裝）
        self.d_input = torch.empty((BATCH, 3, IMGSZ, IMGSZ),
                                   dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return BATCH

    def get_batch(self, names):
        if self.idx + BATCH > len(self.files):
            return None
        batch = np.stack([letterbox(cv2.imread(str(f)))
                          for f in self.files[self.idx:self.idx + BATCH]])
        self.idx += BATCH
        if self.idx % 20 == 0:
            print(f"  校正進度 {self.idx}/{len(self.files)}")
        self.d_input.copy_(self.torch.from_numpy(batch))
        return [int(self.d_input.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache):
            print(f"  沿用既有校正快取 {self.cache}")
            return open(self.cache, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        open(self.cache, "wb").write(cache)
        print(f"  校正快取已寫入 {self.cache}")


def main():
    import torch
    torch.zeros(1, device="cuda")     # 先建立 CUDA context

    files = sorted(CALIB_DIR.glob("*.jpg"))
    if not files:
        raise SystemExit("calib/ 沒有校正影像")
    print(f"校正影像 {len(files)} 張，來自 "
          f"{len({f.name[:13] for f in files})} 個時段")

    builder = trt.Builder(TRT_LOGGER)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, TRT_LOGGER)
    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_flag(trt.BuilderFlag.FP16)         # 不支援 INT8 的層退回 FP16
    cfg.int8_calibrator = Calibrator(files)

    print("開始建置 INT8 engine（會先跑校正，再 profile kernel）...")
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise SystemExit("❌ INT8 engine 建置失敗")
    with open(OUT, "wb") as f:
        f.write(plan)
    print(f"✅ {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
