# Models

Weights are not tracked in git. Download them from the [Releases](../../releases) page into this folder.

| File | What | Size |
|---|---|---|
| `TNC_v11x_0511.pt` | Chicken detector: YOLO11x, 56.9M params, imgsz 960, one class (`TNC`) | 114 MB |

The feeder detector is **not needed at runtime**. The camera is fixed, so feeder boxes were computed once and cached in [`pipeline/config/cached_feeders.json`](../pipeline/config/cached_feeders.json). For a new camera, recalibrate with `research/evaluate_images.py` (feeder step).

**TensorRT engines (`.engine`) are device-specific.** They are compiled for one GPU architecture and TensorRT version, so build them on the target machine:

```bash
python3 pipeline/scripts/export_engine.py --model models/TNC_v11x_0511.pt
```
