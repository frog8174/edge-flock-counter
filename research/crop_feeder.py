# ================================================================
#  Feeder‑only Crop Utility (v2025‑05‑05)
#  ‑ Cut out each feeder ROI from video frames, save crops, and
#    optionally visualize original & enlarged boxes. No secondary
#    TNC prediction.
# ================================================================

from ultralytics import YOLO
import cv2
import os
import pandas as pd


def enlarge_box(x1: float, y1: float, x2: float, y2: float,
                scale: float, img_w: int, img_h: int):
    """Return an enlarged bbox (xyxy). Center preserved, clipped to image."""
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    nw, nh = w * scale, h * scale
    x1n = int(max(cx - nw / 2, 0))
    y1n = int(max(cy - nh / 2, 0))
    x2n = int(min(cx + nw / 2, img_w - 1))
    y2n = int(min(cy + nh / 2, img_h - 1))
    return x1n, y1n, x2n, y2n


def crop_feeder_video(
    feeder_weights: str,
    video_path: str,
    out_root: str = "crop_feeder",
    scale: float = 2.0,
    conf: float = 0.65,
    device: str | int = 0,
):
    """Detect feeders in a video, enlarge each bbox, save crops & visualisations.

    Args:
        feeder_weights: path to feeder YOLO weights.
        video_path:     input video.
        out_root:       base folder for crops and visual results.
        scale:          enlargement factor (>1).
        conf:           feeder detection confidence threshold.
        device:         CUDA index or "cpu".
    """

    assert os.path.exists(video_path), f"Cannot find {video_path}"
    os.makedirs(out_root, exist_ok=True)
    crop_dir = os.path.join(out_root, "crops")    # all crops flat here
    vis_dir = os.path.join(out_root, "box_visualizations")
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load model & video info
    model = YOLO(feeder_weights)
    cap = cv2.VideoCapture(video_path)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stats = []
    global_crop_idx = 0  # continuous crop numbering across frames (optional)

    for frame_idx, r in enumerate(
        model(
            video_path,
            stream=True,
            conf=conf,
            iou=0.5,
            imgsz=(img_h, img_w),
            device=device,
            augment=True,
            verbose=False,
        )
    ):
        frame = r.orig_img
        boxes = r.boxes.xyxy.cpu().numpy()
        vis = frame.copy()
        crop_cnt = 0

        for (x1, y1, x2, y2) in boxes:
            # draw original bbox (green)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # enlarged bbox (red) & crop
            x1n, y1n, x2n, y2n = enlarge_box(x1, y1, x2, y2, scale, img_w, img_h)
            cv2.rectangle(vis, (x1n, y1n), (x2n, y2n), (0, 0, 255), 2)
            crop = frame[y1n:y2n, x1n:x2n]

            # filename: crop_<frame>_<idx>.png  (e.g. crop_0003_01.png)
            crop_path = os.path.join(crop_dir, f"crop_{frame_idx:04d}_{crop_cnt:02d}.png")
            cv2.imwrite(crop_path, crop)
            crop_cnt += 1
            global_crop_idx += 1  # kept for future global indexing if needed

        # save visualisation per frame
        cv2.imwrite(os.path.join(vis_dir, f"frame{frame_idx:04d}.jpg"), vis)
        stats.append({"frame": frame_idx, "feeder_crops": crop_cnt})

    # save csv summary
    df = pd.DataFrame(stats)
    csv_path = os.path.join(out_root, "crop_stats.csv")
    df.to_csv(csv_path, index=False)
    print(df.tail())
    print(f"Summary saved → {csv_path}")


if __name__ == "__main__":
    crop_feeder_video(
        feeder_weights="./models/best_feeder.pt",
        video_path="data/videos/20250504/20250504_1600.mp4",
        scale=2.0,
        conf=0.65,
        device=0,
    )
