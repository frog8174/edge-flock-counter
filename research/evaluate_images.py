# feeder_detect.py
from ultralytics import YOLO
import json, cv2, numpy as np, os
from ultralytics.utils.plotting import Annotator
from sklearn.cluster import KMeans          # ★ 新增：用來做 1-D k-means
from pathlib import Path

FEEDER_MODEL = "./models/feeder.pt"          # 只含 feeder 類別
IMAGE_PATH   = "./images/20250507_rpi2_L/20250507_1806_0000.png"
OUT_JSON     = "./cached_feeders.json"       # 存 bbox 與分級結果
N_GROUPS     = 2                             # 分幾層（固定 2）

model  = YOLO(FEEDER_MODEL)
r      = model(IMAGE_PATH, imgsz=960, conf=0.2, augment=True)[0]

# 取所有 feeder box
feeders = []
for b in r.boxes:
    x1, y1, x2, y2 = map(int, b.xyxy.squeeze().cpu().tolist())
    w, h = x2 - x1, y2 - y1
    cy   = y1 + h / 2                # ← 中心 y 位置
    feeders.append({"bbox": [x1, y1, w, h],
                    "cy":   cy})

# --------------------------------------------------------------------------
# ↓↓↓↓↓          **以下為修改後的分群邏輯 (1-D K-Means)**          ↓↓↓↓↓
# --------------------------------------------------------------------------
ys = np.array([f["cy"] for f in feeders]).reshape(-1, 1)

kmeans = KMeans(n_clusters=N_GROUPS, n_init="auto", random_state=0).fit(ys)
labels = kmeans.labels_                       # 0 或 1

# 讓 y 較小者永遠是 0，較大者永遠是 1
centers = kmeans.cluster_centers_.flatten()
order   = np.argsort(centers)                 # e.g. [1,0] 代表要交換
mapping = {order[0]: 0, order[1]: 1}
labels  = np.vectorize(mapping.get)(labels)   # 重新映射成 0/1

# 寫回 feeder 物件
for f, g in zip(feeders, labels):
    f["group"] = int(g)
# --------------------------------------------------------------------------

# 安全檢查：必須只出現 0 與 1
assert {f["group"] for f in feeders} == {0, 1}, "Grouping error: should be exactly two groups (0,1)."

json.dump(feeders, open(OUT_JSON, "w"))
print(f"[OK] {len(feeders)} feeders stored to {OUT_JSON}")



# chicken_infer.py
# ───────────────────[ 使用者參數 ]───────────────────
CHICK_MODEL = "./models/TNC_v11x_0511.pt"
IMG_DIR     = "data/threshold_countdata/images"
LABEL_DIR   = "data/threshold_countdata/labels"  # ← 新增：YOLO 標註
FEEDER_JSON = "./cached_feeders.json"
OUT_DIR     = "./runs/feeding_detection"
THRESH      = [45.10, 94.23]          # 各組距離閾值 (px) ─ group 0 / group 1
C_THICKNESS = 1                       # 圖上框線粗細
EAT_ID      = 0                      # 標註檔裡「吃飼料」的 class id
IOU_THRESH_EVAL = 0.5                 # 評估用 IOU 門檻
FONT_FACE     = cv2.FONT_HERSHEY_SIMPLEX  # 底標字體
LABEL_OFFSET  = 18  
COL_TP = (0, 255,   0)   # 綠框
COL_FP = (0,   0, 255)   # 紅框
COL_FN = (255,  0,   0)  # 藍框
# ──────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# === 1. 讀 feeder 位置與分組 =============================================
feeders = json.load(open(FEEDER_JSON))
f_cent  = np.array([[b[0]+b[2]/2, b[1]+b[3]/2] for b in [f["bbox"] for f in feeders]])
f_group = [f["group"] for f in feeders]

# === 2. 建立模型 ===========================================================
model = YOLO(CHICK_MODEL)

# === 3. 全域統計器 =========================================================
total_TP = total_FP = total_FN = 0

# === 4. 小工具：計算 IOU ===================================================
def _iou(boxA, boxB):
    """
    boxA, boxB: [x1,y1,x2,y2] (px)
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

# === 5. 主迴圈：逐張影像推論 + 評估 + 畫圖 ================================
for fn in sorted(os.listdir(IMG_DIR)):
    if not fn.lower().endswith((".jpg", ".png")):
        continue
    img_path   = f"{IMG_DIR}/{fn}"
    label_path = f"{LABEL_DIR}/{Path(fn).stem}.txt"

    # ---------- 5-1. 讀取標註 ----------
    gt_eat = []            # ground-truth「吃飼料」盒
    if os.path.exists(label_path):
        for line in open(label_path):
            cls, cx, cy, w, h = map(float, line.split())
            cls = int(cls)
            if cls != EAT_ID:
                continue           # 跳過 Feeder 或其他類別
            # --- 把 YOLO norm 轉成像素 box -----------------
            img = cv2.imread(img_path) if 'img' not in locals() else img
            H, W = img.shape[:2]
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            gt_eat.append([x1, y1, x2, y2])
    gt_matched = [False]*len(gt_eat)

    # ---------- 5-2. 推論 ----------
    pred = model(img_path, conf=0.49, iou=0.60, imgsz=960,
                 device=0, max_det=1000, verbose=False,
                 agnostic_nms=False, augment=True)[0]

    img    = cv2.imread(img_path)        # 已保證有讀
    annot  = Annotator(img, line_width=C_THICKNESS)

    # ---------- 5-3. 畫 feeder ----------
    for f, (cx, cy) in zip(feeders, f_cent):
        x, y, w, h = f["bbox"]
        grp        = f["group"]
        # annot.box_label([x, y, x + w, y + h], f"F{grp}", color=(255,128,0))
        cv2.circle(img, (int(cx), int(cy)), int(THRESH[grp]),
                (255,255,255), C_THICKNESS)

    # ---------- 5-4. 處理每隻雞 ----------
    TP = FP = 0
    for b in pred.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy.squeeze().cpu().tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dists  = np.linalg.norm(f_cent - np.array([cx, cy]), axis=1)
        idx    = int(np.argmin(dists))
        eat    = dists[idx] <= THRESH[f_group[idx]]

        # -------- A. 普通雞 (non-eat) ------------
        if not eat:
            annot.box_label([x1, y1, x2, y2],          # ← 紅框標「chick」
                            "chick", color=(153, 153, 255))
            continue

        # -------- B. 吃飼料 → 判斷 TP / FP -------
        matched = False
        for gi, gbox in enumerate(gt_eat):
            if gt_matched[gi]:          # ground-truth 已配對
                continue
            if _iou([x1, y1, x2, y2], gbox) >= IOU_THRESH_EVAL:
                matched = gt_matched[gi] = True
                break

        if matched:
            # ★★★ 這裡畫 TP 盒 (綠) ★★★
            annot.box_label([x1, y1, x2, y2],
                            "TP", color=(0, 255, 0), txt_color=(0, 0, 0))
            TP += 1
        else:
            # ★★★ 這裡畫 FP 盒 (紅) ★★★
            annot.box_label([x1, y1, x2, y2],
                            "FP", color=(0, 0, 255), txt_color=(255, 255, 255))
            FP += 1
        
    # ---------- 5-5. 找 FN ----------
    unmatched_idx = [gi for gi, m in enumerate(gt_matched) if not m]  # 尚未配對的 GT

    for gi in unmatched_idx:
        x1, y1, x2, y2 = map(int, gt_eat[gi])
        annot.box_label([x1, y1, x2, y2],          # 藍框 “FN”
                        "FN", color=(255, 0, 0), txt_color=(0, 0, 0))

    # 只算一次
    FN = len(unmatched_idx)

    # ---------- 5-6. 統計 + 畫文字 ----------
    precision = TP / (TP + FP) if TP+FP else 0
    recall    = TP / (TP + FN) if TP+FN else 0
    total_TP += TP; total_FP += FP; total_FN += FN

    # 在影像下加白底條
    eat_total = TP + FP + FN
    H, W = img.shape[:2]
    canvas = np.full((H+60, W, 3), 255, dtype=np.uint8)
    canvas[:H] = annot.result()
    txt = f"P={precision:.3f}  R={recall:.3f}  Eat={eat_total}  TP={TP}  FP={FP}  FN={FN}"
    cv2.putText(canvas, txt, (15, H+48), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0,0,0), 3, cv2.LINE_AA)

    out_path = f"{OUT_DIR}/{fn}"
    cv2.imwrite(out_path, canvas)
    print(f"[{fn}] {txt}")

# === 6. 總結結果 ===========================================================
tot_prec = total_TP / (total_TP + total_FP) if total_TP+total_FP else 0
tot_rec  = total_TP / (total_TP + total_FN) if total_TP+total_FN else 0
tot_f1   = (2*tot_prec*tot_rec/(tot_prec+tot_rec)) if tot_prec+tot_rec else 0

print("\n================ Overall =================")

print(f"Precision={tot_prec:.4f}  Recall={tot_rec:.4f}  F1={tot_f1:.4f}")