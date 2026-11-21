#!/usr/bin/env python3
"""產出可視化的推論結果：三種配置在同一幀上的對比。"""
import gc, json
import cv2, numpy as np, torch
from ultralytics import YOLO
from trt_runner import TRTRunner

VIDEO="data/videos/20250501/20250501_0700.mp4"
FRAME_IDX=40
IMGSZ,CONF,IOU,MD=960,0.49,0.60,1000
FEEDERS=json.load(open('pipeline/config/cached_feeders.json'))
THRESH=[45.10,94.23]

cap=cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES,FRAME_IDX)
ok,frame=cap.read(); cap.release()
assert ok

def ultra(path,aug):
    m=YOLO(path)
    for _ in range(2): m.predict(frame,imgsz=IMGSZ,conf=CONF,iou=IOU,max_det=MD,augment=aug,device=0,verbose=False)
    r=m.predict(frame,imgsz=IMGSZ,conf=CONF,iou=IOU,max_det=MD,augment=aug,device=0,verbose=False)[0]
    b=r.boxes.xyxy.cpu().numpy(); del m; gc.collect(); torch.cuda.empty_cache(); return b

def native(eng):
    r=TRTRunner(eng,imgsz=IMGSZ,conf=CONF,iou=IOU,max_det=MD)
    for _ in range(2): r.infer(frame)
    b,_=r.infer(frame); b=b.cpu().numpy(); del r; gc.collect(); torch.cuda.empty_cache(); return b

print("推論中...")
runs=[("A  PyTorch FP32 + TTA","thesis baseline", ultra("models/TNC_v11x_0511.pt",True),   (60,180,255)),
      ("C  TensorRT FP16","adopted",        ultra("models/TNC_v11x_0511.engine",False),(80,220,80)),
      ("E  TensorRT INT8","accuracy collapse",        native("models/TNC_int8.engine"),          (70,70,235))]
for t,_,b,_c in runs: print(f"  {t}: {len(b)} 隻")

f_cent=np.array([[f["bbox"][0]+f["bbox"][2]/2, f["bbox"][1]+f["bbox"][3]/2] for f in FEEDERS])
f_rad=np.array([THRESH[f["group"]] for f in FEEDERS])

def draw(boxes,color,label,sub):
    im=frame.copy()
    for (cx,cy),rad in zip(f_cent,f_rad):
        cv2.circle(im,(int(cx),int(cy)),int(rad),(255,255,255),2)
    for x1,y1,x2,y2 in boxes.astype(int):
        cv2.rectangle(im,(x1,y1),(x2,y2),color,2)
    bar=np.full((150,im.shape[1],3),28,dtype=np.uint8)
    cv2.putText(bar,label,(30,58),cv2.FONT_HERSHEY_SIMPLEX,1.7,(245,245,245),4)
    cv2.putText(bar,f"{len(boxes)} detections",(30,118),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,4)
    cv2.putText(bar,sub,(im.shape[1]-560,118),cv2.FONT_HERSHEY_SIMPLEX,1.2,(150,150,150),3)
    return np.vstack([bar,im])

# ── 全幀三欄對比 ──
panels=[draw(b,c,t,s) for t,s,b,c in runs]
full=np.hstack(panels)
full=cv2.resize(full,(2880,int(full.shape[0]*2880/full.shape[1])),interpolation=cv2.INTER_AREA)
cv2.imwrite("compare_full.jpg",full,[cv2.IMWRITE_JPEG_QUALITY,88])
print(f"→ compare_full.jpg  {full.shape[1]}×{full.shape[0]}")

# ── 放大局部：挑一個近排餵食器周邊 ──
g1=[i for i,f in enumerate(FEEDERS) if f["group"]==1]
ci=g1[1]; cx,cy=f_cent[ci]
W,H=560,420
x0,y0=int(cx-W//2),int(cy-H//2)
x0=max(0,min(x0,frame.shape[1]-W)); y0=max(0,min(y0,frame.shape[0]-H))

def crop_panel(boxes,color,label):
    im=frame.copy()
    cv2.circle(im,(int(cx),int(cy)),int(f_rad[ci]),(255,255,255),2)
    for x1,y1,x2,y2 in boxes.astype(int):
        cv2.rectangle(im,(x1,y1),(x2,y2),color,2)
    c=im[y0:y0+H,x0:x0+W]
    c=cv2.resize(c,(W*2,H*2),interpolation=cv2.INTER_CUBIC)
    n=sum(1 for x1,y1,x2,y2 in boxes if x0<=(x1+x2)/2<x0+W and y0<=(y1+y2)/2<y0+H)
    bar=np.full((92,c.shape[1],3),28,dtype=np.uint8)
    cv2.putText(bar,label,(22,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(245,245,245),2)
    cv2.putText(bar,f"{n} in view",(22,78),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
    return np.vstack([bar,c])

zoom=np.hstack([crop_panel(b,c,t) for t,s,b,c in runs])
cv2.imwrite("compare_zoom.jpg",zoom,[cv2.IMWRITE_JPEG_QUALITY,92])
print(f"→ compare_zoom.jpg  {zoom.shape[1]}×{zoom.shape[0]}")
