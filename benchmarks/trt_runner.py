#!/usr/bin/env python3
"""
原生 TensorRT 推論器 — 不經過 ultralytics。

存在的兩個理由：
  1. DLA engine（trtexec 產出）沒有 ultralytics 的 metadata 標頭，它的載入器讀不了
  2. 拿掉 ultralytics 的 Python 包裝開銷，這正是先前診斷出的瓶頸之一

前處理用 letterbox 對齊訓練時的設定；NMS 用 torchvision 在 GPU 上做，
避免 650+ 目標的 NMS 回到 CPU。
"""
import numpy as np
import tensorrt as trt
import torch
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def _strip_ultralytics_header(blob: bytes) -> bytes:
    """ultralytics 會在 engine 前面塞 4-byte 長度 + JSON metadata。裸 engine 則沒有。"""
    if len(blob) < 4:
        return blob
    n = int.from_bytes(blob[:4], "little")
    if 0 < n < 4096 and blob[4:4 + n].lstrip()[:1] in (b"{", b'"'):
        return blob[4 + n:]
    return blob


class TRTRunner:
    def __init__(self, engine_path, imgsz=960, conf=0.49, iou=0.60, max_det=1000):
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        with open(engine_path, "rb") as f:
            blob = _strip_ultralytics_header(f.read())
        rt = trt.Runtime(TRT_LOGGER)
        self.engine = rt.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"無法反序列化 {engine_path}")
        self.ctx = self.engine.create_execution_context()

        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name, self.in_shape = n, tuple(self.engine.get_tensor_shape(n))
            else:
                self.out_name, self.out_shape = n, tuple(self.engine.get_tensor_shape(n))

        self.d_in = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")
        self.d_out = torch.empty(self.out_shape, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.in_name, self.d_in.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.d_out.data_ptr())
        self.stream = torch.cuda.Stream()

    # ── 前處理：letterbox → RGB → CHW → /255，全程在 GPU ──
    def preprocess(self, img):
        h, w = img.shape[:2]
        s = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * s)), int(round(w * s))
        if (nh, nw) != (h, w):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = img
        t = torch.from_numpy(canvas).cuda().permute(2, 0, 1).flip(0).float().div_(255.0)
        self.d_in.copy_(t.unsqueeze(0))
        return s, left, top

    def infer(self, img):
        s, left, top = self.preprocess(img)
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # 輸出 (1, 4+nc, anchors)；單類別 → (1, 5, N)
        o = self.d_out[0]                      # (5, N)
        scores = o[4]
        keep = scores > self.conf
        if keep.sum() == 0:
            return torch.empty((0, 4)), torch.empty((0,))
        b = o[:4, keep].T                      # (M, 4) cx,cy,w,h
        scores = scores[keep]

        xy, wh = b[:, :2], b[:, 2:] / 2
        boxes = torch.cat([xy - wh, xy + wh], dim=1)          # xyxy (letterbox 座標系)
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= s                                             # 還原到原圖座標

        from torchvision.ops import nms
        idx = nms(boxes, scores, self.iou)[: self.max_det]
        return boxes[idx], scores[idx]
