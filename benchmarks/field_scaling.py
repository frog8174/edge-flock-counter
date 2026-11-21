#!/usr/bin/env python3
"""
場域擴充性推算 — 從實測的單路/雙路數據，推算全場 150 台相機的部署需求。

依簡報 slide 7 / 23 的雞舍配置：
    單棟 113m × 14.8m，相機間距 8m → 每棟 15 台；共 10 棟 → 全場 150 台
"""
import json
from pathlib import Path

L, SPACING, ROWS = 113.0, 8.0, 10
CAM_PER_HOUSE = int(L // SPACING) + 1
TOTAL_CAM = CAM_PER_HOUSE * ROWS
REALTIME_FPS = 5.0            # 現場相機幀率 = 即時門檻
DAY_HOURS = 13                # 影片涵蓋 0700–1954
BITRATE = 2.0e6               # H.264 ~2 Mbps
FEED_WEEKS = 12

RPI_IDLE_W = 3.0              # Raspberry Pi 4B 錄影時約略功耗


def video_gb_per_cam_day():
    return BITRATE * 3600 * DAY_HOURS / 8 / 1e9


def analyse(results):
    """results: stream_bench.py 產出的 list[dict]"""
    out = {
        "cam_per_house": CAM_PER_HOUSE, "houses": ROWS, "total_cam": TOTAL_CAM,
        "realtime_fps": REALTIME_FPS,
        "video_gb_cam_day": round(video_gb_per_cam_day(), 2),
        "video_tb_field_feedperiod": round(
            video_gb_per_cam_day() * TOTAL_CAM * FEED_WEEKS * 7 / 1000, 1),
        "configs": [],
    }

    for r in results:
        n = r["n_streams"]
        agg = r["agg_fps"]
        min_fps = r["min_stream_fps"]
        power = r.get("power_w")

        # 這個設定下，一台 Jetson 在維持即時的前提下能吃幾路
        capacity = agg / REALTIME_FPS
        ok = r["realtime_ok"]

        cfg = {
            "n_streams": n,
            "agg_fps": agg,
            "min_stream_fps": min_fps,
            "realtime_ok": ok,
            "power_w": power,
            "fps_per_watt": r.get("fps_per_watt"),
            "gpu_util_pct": r.get("gpu_util_pct"),
            "streams_at_realtime": round(capacity, 2),
            "scaling_efficiency": None,
        }
        out["configs"].append(cfg)

    # 擴充效率：雙路聚合吞吐 ÷ (單路吞吐 × 2)
    by_n = {c["n_streams"]: c for c in out["configs"]}
    if 1 in by_n and 2 in by_n:
        s1, s2 = by_n[1]["agg_fps"], by_n[2]["agg_fps"]
        by_n[2]["scaling_efficiency"] = round(s2 / (s1 * 2), 3)

    # 全場部署推算：用實測中「仍維持即時」的最大路數
    viable = [c for c in out["configs"] if c["realtime_ok"]]
    if viable:
        best = max(viable, key=lambda c: c["n_streams"])
        per_box = best["n_streams"]
        boxes = -(-TOTAL_CAM // per_box)          # ceil
        out["deployment"] = {
            "verified_streams_per_box": per_box,
            "jetsons_needed": boxes,
            "rpi_replaced": TOTAL_CAM,
            "reduction_ratio": round(TOTAL_CAM / boxes, 1),
            "field_power_jetson_w": round(boxes * (best["power_w"] or 0), 1),
            "field_power_rpi_w": round(TOTAL_CAM * RPI_IDLE_W, 1),
        }
    return out


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results.json")
    res = json.loads(src.read_text())
    a = analyse(res)
    print(json.dumps(a, indent=2, ensure_ascii=False))
    Path("field_scaling.json").write_text(json.dumps(a, indent=2, ensure_ascii=False))
    print("\n→ field_scaling.json")
