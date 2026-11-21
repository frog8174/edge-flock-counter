#!/usr/bin/env python3
"""
混合精度 INT8 — 只量化 backbone 卷積，敏感層保留 FP16。

純 INT8 的量測結果：1.62× 加速，但 recall 只剩 36.7%（precision 91%）。
降低 conf 門檻也救不回來（conf=0.05 時 recall 僅 53.6%、precision 崩到 39.8%），
代表是特徵提取被量化破壞，不是信心分數被壓縮。

診斷指向 YOLO11x 的 C2PSA 注意力區塊：softmax 與 matmul 的動態範圍很寬，
是 INT8 量化的重災區。偵測頭的 dfl/Softmax 同理。

策略：這些層強制 FP16，其餘走 INT8。校正快取沿用（int8_calib.cache）。
"""
import os
import tensorrt as trt
from build_int8 import Calibrator, CALIB_DIR, ONNX

OUT = "models/TNC_int8_mixed.engine"
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# 量化敏感、強制保留 FP16 的層
SENSITIVE_TYPES = {trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.SOFTMAX}
SENSITIVE_NAME_PARTS = ("model.23",)          # 偵測頭
SENSITIVE_BLOCKS = ("model.10",)              # C2PSA 注意力區塊


def main():
    import torch
    torch.zeros(1, device="cuda")

    from pathlib import Path
    files = sorted(Path(CALIB_DIR).glob("*.jpg"))

    builder = trt.Builder(TRT_LOGGER)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, TRT_LOGGER)
    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors): print(parser.get_error(i))
            raise SystemExit(1)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    cfg.int8_calibrator = Calibrator(files)

    forced = 0
    for i in range(net.num_layers):
        l = net.get_layer(i)
        # CONSTANT / 整數型別的層不能設浮點精度（權重可能是 Int64）
        if l.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE,
                      trt.LayerType.GATHER, trt.LayerType.CAST):
            continue
        hit = (l.type in SENSITIVE_TYPES
               or any(p in l.name for p in SENSITIVE_NAME_PARTS)
               or any(p in l.name for p in SENSITIVE_BLOCKS))
        if hit and all(l.get_output(j).dtype in (trt.float32, trt.float16)
                       for j in range(l.num_outputs)):
            l.precision = trt.float16
            for j in range(l.num_outputs):
                l.set_output_type(j, trt.float16)
            forced += 1

    print(f"總層數 {net.num_layers}，強制 FP16 {forced} 層 "
          f"({forced/net.num_layers*100:.1f}%)，其餘走 INT8")
    print("建置中（校正快取已存在，跳過重新校正）...")

    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise SystemExit("❌ 建置失敗")
    open(OUT, "wb").write(plan)
    print(f"✅ {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
