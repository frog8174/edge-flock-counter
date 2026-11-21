#!/usr/bin/env python3
import os, glob, cv2, json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from scipy.optimize import linear_sum_assignment  # Hungarian

# =========  路徑 & 參數  =========
LABEL_DIR   = "data/threshold_countdata/labels"
IMAGE_DIR   = "data/threshold_countdata/images"
CHICK_MODEL = "./models/TNC_v11x_0511.pt"
THRESH      = [45.10, 94.23]
IOU_THRESH  = 0.4
FEEDER_ID, CHICK_ID = 1, 0
# =================================

def iou(boxA, boxB):
    # box = (x1,y1,x2,y2)
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

def center(box): x1,y1,x2,y2=box; return np.array([(x1+x2)/2,(y1+y2)/2])

# ---------- 解析標註 ----------
feeders, chicks = defaultdict(list), defaultdict(list)
txts = glob.glob(os.path.join(LABEL_DIR,"**","*.txt"),recursive=True)
for p in txts:
    name = os.path.splitext(os.path.basename(p))[0]
    h,w = cv2.imread(f"{IMAGE_DIR}/{name}.png").shape[:2]
    with open(p) as f:
        for line in f:
            c,cx,cy,bw,bh = map(float,line.split())
            x1 = (cx-bw/2)*w; y1=(cy-bh/2)*h
            x2 = (cx+bw/2)*w; y2=(cy+bh/2)*h
            if int(c)==FEEDER_ID:  feeders[name].append((x1,y1,x2,y2))
            elif int(c)==CHICK_ID: chicks[name].append((x1,y1,x2,y2))

# ---------- 推論模型 ----------
model = YOLO(CHICK_MODEL)

y_true, y_pred = [], []

for name in chicks:
    img_path = f"{IMAGE_DIR}/{name}.png"
    if not os.path.exists(img_path): continue
    gt_feed = np.array(feeders[name]); gt_ch = chicks[name]
    if len(gt_feed)==0 or len(gt_ch)==0: continue

    # === feeder 分群 (KMeans on y) ===
    ys = gt_feed[:,1::2].mean(axis=1).reshape(-1,1)
    kmeans = KMeans(n_clusters=2,n_init="auto").fit(ys)
    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers); mapping = {order[0]:0,order[1]:1}
    f_groups = [mapping[l] for l in kmeans.labels_]

    # === 計算 gt eating 標籤 ===
    f_cent = np.array([center(b) for b in gt_feed])
    gt_eat_flags=[]
    for box in gt_ch:
        d = np.linalg.norm(f_cent - center(box),axis=1)
        idx=int(np.argmin(d)); gt_eat_flags.append(d[idx]<=THRESH[f_groups[idx]])

    # === 預測 ===
    r = model(img_path,conf=0.49,iou=0.6,imgsz=960,verbose=False)[0]
    pred_boxes = [tuple(map(float,b.xyxy.squeeze().cpu().tolist())) for b in r.boxes]
    pred_flags=[]
    for b in r.boxes:
        x1,y1,x2,y2=map(float,b.xyxy.squeeze().cpu().tolist())
        d = np.linalg.norm(f_cent - np.array([(x1+x2)/2,(y1+y2)/2]),axis=1)
        idx=int(np.argmin(d)); pred_flags.append(d[idx]<=THRESH[f_groups[idx]])

    # === 配對 Pred↔GT (IoU) ===
    if pred_boxes:
        cost = [[1-iou(pb,gb) for pb in pred_boxes] for gb in gt_ch]
        row_ind,col_ind = linear_sum_assignment(cost)
        matched=set(); gt_matched=set()
        for gi,pi in zip(row_ind,col_ind):
            if iou(pred_boxes[pi],gt_ch[gi])>=IOU_THRESH:
                y_true.append(gt_eat_flags[gi])
                y_pred.append(pred_flags[pi])
                matched.add(pi); gt_matched.add(gi)
        # 未配對 pred → FP (non-eat=False)
        for i,p in enumerate(pred_boxes):
            if i not in matched:
                y_true.append(False)   # 沒有對應 gt，視為 non-eat
                y_pred.append(pred_flags[i])
        # 未配對 gt → FN
        for gi in range(len(gt_ch)):
            if gi not in gt_matched:
                y_true.append(gt_eat_flags[gi])
                y_pred.append(False)   # 沒偵測到
    else:
        # 沒任何預測 → 全 FN
        y_true.extend(gt_eat_flags)
        y_pred.extend([False]*len(gt_eat_flags))
    out_dir = "viz_eval"
    os.makedirs(out_dir, exist_ok=True)
    canvas = cv2.imread(img_path).copy()

    # ❶ 畫 feeder 方便對照
    for (x1,y1,x2,y2), grp in zip(gt_feed, f_groups):
        cv2.rectangle(canvas, (int(x1),int(y1)), (int(x2),int(y2)),
                    (255,128,0), 1)
        cv2.putText(canvas, f"F{grp}", (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,0), 1, cv2.LINE_AA)

    # ❷ 先記錄各類型 index -------------------------------------------------
    #    matched： (gi,pi)  → 已在上面配對成功
    tp_idx, fp_idx, fn_idx = [], [], []

    for gi,pi in zip(row_ind,col_ind):
        if iou(pred_boxes[pi],gt_ch[gi])<IOU_THRESH:   # 其實配對已過濾，但保險
            continue
        gt_eat = gt_eat_flags[gi]
        pred_eat = pred_flags[pi]
        if gt_eat and pred_eat:
            tp_idx.append((gi,pi))
        elif (not gt_eat) and pred_eat:
            fp_idx.append((gi,pi))
        #  gt non-eat & pred non-eat 不標
    for gi in range(len(gt_ch)):
        if gi not in gt_matched and gt_eat_flags[gi]:
            fn_idx.append(gi)

    # ❸ 畫 TP / FP / FN -----------------------------------------------------
    for gi,pi in tp_idx:
        x1,y1,x2,y2 = map(int, pred_boxes[pi])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(canvas,"TP",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (0,255,0),1,cv2.LINE_AA)

    for gi,pi in fp_idx:
        x1,y1,x2,y2 = map(int, pred_boxes[pi])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.putText(canvas,"FP",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (0,255,255),1,cv2.LINE_AA)

    for gi in fn_idx:
        x1,y1,x2,y2 = map(int, gt_ch[gi])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(canvas,"FN",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (0,0,255),1,cv2.LINE_AA)

    # ❹ 在底部加「統計條」 -----------------------------------------------
    pred_eat  = sum(pred_flags)
    label_eat = sum(gt_eat_flags)
    tp_eat    = len(tp_idx)
    precision = tp_eat / pred_eat  if pred_eat  else 0.0
    recall    = tp_eat / label_eat if label_eat else 0.0

    text = (f"Predict EAT: {pred_eat}   "
            f"Label EAT: {label_eat}   "
            f"Precision: {precision:.2f}   Recall: {recall:.2f}")

    # 先在畫面底部開一條黑底
    h, w = canvas.shape[:2]
    pad  = 30
    canvas = cv2.copyMakeBorder(canvas, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    cv2.putText(canvas, text, (10, h + pad - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # ❺ 輸出
    cv2.imwrite(f"{out_dir}/{name}.jpg", canvas)
print(classification_report(y_true, y_pred, target_names=["non-EAT","EAT"]))
