# feeder_detect.py
from ultralytics import YOLO
import json, cv2, numpy as np, os
from ultralytics.utils.plotting import Annotator
from sklearn.cluster import KMeans          # ★ 新增：用來做 1-D k-means

FEEDER_MODEL = "./models/feeder.pt"         # 只含 feeder 類別
IMAGE_PATH   = "./images/20250507_rpi2_L/20250507_1806_0000.png"
OUT_JSON     = "./cached_feeders.json"      # 存 bbox 與分級結果
N_GROUPS     = 2                            # 分幾層（固定 2）

model   = YOLO(FEEDER_MODEL)
r       = model(IMAGE_PATH, imgsz=960, conf=0.2, augment=True)[0]

# 取所有 feeder box
feeders = []
for b in r.boxes:
    x1, y1, x2, y2 = map(int, b.xyxy.squeeze().cpu().tolist())
    w, h = x2 - x1, y2 - y1
    cy   = y1 + h / 2                       # ← 中心 y 位置
    feeders.append({"bbox": [x1, y1, w, h],
                    "cy":   cy})

# --------------------------------------------------------------------------
# ↓↓↓↓↓         **以下為修改後的分群邏輯 (1-D K-Means)** ↓↓↓↓↓
# --------------------------------------------------------------------------
ys = np.array([f["cy"] for f in feeders]).reshape(-1, 1)

kmeans = KMeans(n_clusters=N_GROUPS, n_init="auto", random_state=0).fit(ys)
labels = kmeans.labels_                         # 0 或 1

# 讓 y 較小者永遠是 0，較大者永遠是 1
centers = kmeans.cluster_centers_.flatten()
order   = np.argsort(centers)                    # e.g. [1,0] 代表要交換
mapping = {order[0]: 0, order[1]: 1}
labels  = np.vectorize(mapping.get)(labels)      # 重新映射成 0/1

# 寫回 feeder 物件
for f, g in zip(feeders, labels):
    f["group"] = int(g)
# --------------------------------------------------------------------------

# 安全檢查：必須只出現 0 與 1
assert {f["group"] for f in feeders} == {0, 1}, "Grouping error: should be exactly two groups (0,1)."

json.dump(feeders, open(OUT_JSON, "w"))
print(f"[OK] {len(feeders)} feeders stored to {OUT_JSON}")


# ========= 可自行修改 =========
CHICK_MODEL = "./models/TNC_v11x_0511.pt"
FEEDER_JSON = "./cached_feeders.json"
VIDEO_IN    = "./Videos/20250507_1600.mp4"      # ← 輸入影片
OUT_VIDEO   = "./runs/feeding_detection/20250507_1600_EatCount.mp4"
THRESH      = [45.10, 92.70]                   # 各組距離閾值 (px): group 0 / 1
C_THICKNESS = 1                                # 圓與框線條粗細
DEVICE_ID   = 0                                # GPU id (改成 -1 用 CPU)
# ===============================

# 讀取固定 feeder 位置（由 feeder_detect.py 產出）
feeders = json.load(open(FEEDER_JSON))
f_cent  = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2]
                    for b in (f["bbox"] for f in feeders)])
f_group = [f["group"] for f in feeders]

# 初始化 YOLO 模型
model = YOLO(CHICK_MODEL)

# ----------- 打開影片 ------------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"❌ 無法打開影片：{VIDEO_IN}")

fps     = cap.get(cv2.CAP_PROP_FPS)
w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)
writer  = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w_frame, h_frame))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推論（傳入 numpy BGR 圖即可）
    r = model(frame, conf=0.49, iou=0.60, imgsz=960,
              device=DEVICE_ID, max_det=1000, verbose=False,
              agnostic_nms=False, augment=False)[0]

    annot = Annotator(frame, line_width=C_THICKNESS)

    # --- 畫 feeder 框 + radius 圓 -------------------------------------
    for f, (cx, cy) in zip(feeders, f_cent):
        x, y, w, h = f["bbox"]
        grp = f["group"]
        annot.box_label([x, y, x + w, y + h], f"F{grp}", color=(255, 128, 0))

        radius = int(THRESH[grp])              # 該組 threshold
        cv2.circle(frame, (int(cx), int(cy)), radius,
                   (0, 0, 0), C_THICKNESS)
    # -----------------------------------------------------------------

    # -------- 判斷每隻雞是否進食 ------------------------------------
    eat_count = 0
    total_count = len(r.boxes)
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy.squeeze().cpu().tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        dists = np.linalg.norm(f_cent - np.array([cx, cy]), axis=1)
        idx   = int(np.argmin(dists))
        is_eating = dists[idx] <= THRESH[f_group[idx]]

        if is_eating:
            eat_count += 1
            label, color = "Around", (0, 255, 0)
        else:
            label, color = "chick", (153, 153, 255)

        annot.box_label([x1, y1, x2, y2], label, color=color)
    # -----------------------------------------------------------------
    
    # -------- vvvvvv 以下為修改區域 vvvvvv ---------------------------
    # -------- 計算比例並顯示在畫面上 -------------------------------
    if total_count > 0:
        ratio = eat_count / total_count * 100
    else:
        ratio = 0.0

    text = f"Around Feeder rate: {ratio:.1f}% ({eat_count}/{total_count})"
    
    # ☆ 字體放大、加粗
    font_scale = 1.4
    thickness = 2
    
    # 在畫面左上角寫字
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=thickness
    )
    # ☆ 調整 y 座標，讓文字與頂部保留更多空間
    x, y = 5, 1088 - text_height
    rect_tl = (x, y - text_height - baseline)
    rect_br = (x + text_width, y + baseline)

    # 畫白底矩形
    cv2.rectangle(frame, rect_tl, rect_br, color=(255, 255, 255), thickness=-1)

    # 再畫文字
    cv2.putText(frame, text,
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=thickness)
    # -------- ^^^^^^ 以上為修改區域 ^^^^^^ ---------------------------
    # -----------------------------------------------------------------

    writer.write(annot.result())
    
    if frame_idx % 100 == 0:
        print(f"processed frame {frame_idx}")
    if frame_idx == 50: break
    frame_idx += 1

# -------- 關閉 I/O ------------
cap.release()
writer.release()
print(f"\n[OK] Done! 影片已輸出至：{OUT_VIDEO}")