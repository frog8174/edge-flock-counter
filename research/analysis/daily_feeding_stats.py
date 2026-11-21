#!/usr/bin/env python3
# daily_feeding_stats.py
# -----------------------------------------------------------
#  讀取一整天 (多段 5min) 影片，以 ~5 fps 推論，
#  統計「總雞數」「進食數」「進食比例」寫入 CSV。
# -----------------------------------------------------------
import os, glob, json, cv2, csv, datetime
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from sklearn.cluster import KMeans        # feeder 分群仍採 1-D k-means
# ★ 新增：用來存逐影格統計
from collections import defaultdict

# ========= 使用者可調參數 ========================================
DAY_DIR        = "data/videos/fullday/"      # 那一天全部 5 分鐘 mp4 的資料夾
FEEDER_MODEL   = "./models/feeder.pt"
CHICK_MODEL    = "./models/TNC_v11x_0511.pt"
OUT_DIR        = "./runs/feeding_detection/20250507_fullday"  # 標註影片與 CSV 皆放這
IMG_SIZE       = 960
CONF_THRES     = 0.49
IOU_THRES      = 0.60
DEVICE_ID      = 0                        # GPU id；設 -1 改用 CPU
FPS_TARGET     = 5                        # 只處理 ≈5 fps
N_GROUPS       = 2                        # feeder 分層 (=飲水層/飼料層)
THRESH         = [45.10, 92.70]           # (px) 兩層 radius
C_THICKNESS    = 1                        # 畫框線寬
# ===============================================================

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Part 1. 偵測 feeder 取得固定座標 ----------------
IMAGE_PATH   = "./images/20250507_rpi2_L/20250507_1806_0000.png"
print("🔍 Detecting feeders …")
yolo_feed = YOLO(FEEDER_MODEL)
r0 = yolo_feed(IMAGE_PATH, imgsz=IMG_SIZE, conf=0.2, augment=True)[0]
feeders = []
for b in r0.boxes:
    x1, y1, x2, y2 = map(int, b.xyxy.squeeze().cpu().tolist())
    w, h = x2 - x1, y2 - y1
    cy   = y1 + h / 2
    feeders.append({"bbox": [x1, y1, w, h], "cy": cy})

# --- 1-D k-means 把 feeder 分成上下兩層 --------------------------
ys = np.array([f["cy"] for f in feeders]).reshape(-1, 1)
kmeans  = KMeans(n_clusters=N_GROUPS, n_init="auto", random_state=0).fit(ys)
centers = kmeans.cluster_centers_.flatten()
order   = np.argsort(centers)                       # 確保上層標 0、下層標 1
mapping = {order[0]: 0, order[1]: 1}
labels  = np.vectorize(mapping.get)(kmeans.labels_)

for f, g in zip(feeders, labels):
    f["group"] = int(g)

assert {f["group"] for f in feeders} == {0, 1}, "Feeder grouping error"
# 轉成 numpy 方便距離運算
f_cent  = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2]
                    for b in (f["bbox"] for f in feeders)])
f_group = [f["group"] for f in feeders]
print(f"✅ {len(feeders)} feeders 分為兩層完成")

# ---------------- Part 2. 初始化 chick 模型 ----------------------
yolo_chick = YOLO(CHICK_MODEL)

# ---------------- Part 3. 逐檔影片推論並統計 ---------------------
csv_path = os.path.join(OUT_DIR, "feeding_stats.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["video", "frame_idx", "timestamp_sec",
                    "total_chicks", "eat_count", "eat_ratio"])

    for vid_path in sorted(glob.glob(os.path.join(DAY_DIR, "*.mp4"))):
        print(f"\n▶ 處理 {os.path.basename(vid_path)}")
        cap   = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  ⚠ 無法開啟影片，略過：{vid_path}")
            continue

        fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30
        skip_n   = max(1, round(fps_in / FPS_TARGET))   # 幀跳讀間隔
        w_frame  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_frame  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_name = os.path.splitext(os.path.basename(vid_path))[0] + "_ED.mp4"
        writer_v = cv2.VideoWriter(os.path.join(OUT_DIR, out_name),
                                fourcc, FPS_TARGET, (w_frame, h_frame))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 只處理每 skip_n 幀
            if frame_idx % skip_n != 0:
                frame_idx += 1
                continue

            # ---- Chick detection ----------------------------------------
            r = yolo_chick(frame, conf=CONF_THRES, iou=IOU_THRES,
                        imgsz=IMG_SIZE, device=DEVICE_ID,
                        max_det=1000, verbose=False,
                        agnostic_nms=False, augment=False)[0]

            annot = Annotator(frame, line_width=C_THICKNESS)

            # 畫 feeder 框 + radius
            for f, (cx, cy) in zip(feeders, f_cent):
                x, y, w, h = f["bbox"]
                grp = f["group"]
                annot.box_label([x, y, x + w, y + h], f"F{grp}",
                                color=(255, 128, 0))
                cv2.circle(frame, (int(cx), int(cy)), int(THRESH[grp]),
                        (255, 255, 255), C_THICKNESS)

            # ---- 判斷每隻雞是否進食 ----------------------------------
            total_chicks = len(r.boxes)
            eat_count    = 0
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy.squeeze().cpu().tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dists  = np.linalg.norm(f_cent - np.array([cx, cy]), axis=1)
                idx    = int(np.argmin(dists))
                eat    = dists[idx] <= THRESH[f_group[idx]]
                eat_count += int(eat)
                annot.box_label([x1, y1, x2, y2],
                                "EAT" if eat else "chick",
                                color=(0, 255, 0) if eat else (0, 0, 255))

            eat_ratio = round(eat_count / total_chicks, 4) if total_chicks else 0
            timestamp = frame_idx / fps_in

            # ---- 寫入 CSV ---------------------------------------------
            writer.writerow([os.path.basename(vid_path), frame_idx,
                            round(timestamp, 2),
                            total_chicks, eat_count, eat_ratio])

            writer_v.write(annot.result())
            frame_idx += 1

        cap.release()
        writer_v.release()
        print(f"  ✔ 完成：{out_name}")

print(f"\n🎉  全日影片處理完畢，統計已存：{csv_path}")
