#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate per-group distance thresholds between chickens and feeders.

變動重點
──────
1. feeder 分群：y-center 1-D K-Means（k = 2）
2. threshold：IQR 去離群 + 90th percentile
3. 繪圖：新增 group_box_density.png（箱形 + 常態密度）
"""
import cv2, os, glob
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")             # 若在無顯示卡伺服器，避免 backend 問題
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde   # ← 新增，用於密度估計

# ========= 可調參數 =========
LABEL_DIR   = "data/threshold_countdata/labels"
IMAGE_DIR   = "data/threshold_countdata/images"
N_GROUPS    = 2
FEEDER_ID   = 1      # feeder class id
CHICKEN_ID  = 0      # chicken class id
# ===========================

feeders, chickens = defaultdict(list), defaultdict(list)
orig_data, clean_data = [], []      # ← 收集兩版本距離

# ---------- 讀取標註 ----------
txt_files = glob.glob(os.path.join(LABEL_DIR, "**", "*.txt"), recursive=True)
if not txt_files:
    raise RuntimeError(f"❌ 找不到任何 txt：{LABEL_DIR}")

for txt_path in txt_files:
    img_name = os.path.splitext(os.path.basename(txt_path))[0]
    img_path = next((p for p in [f"{IMAGE_DIR}/{img_name}.jpg",
                                 f"{IMAGE_DIR}/{img_name}.png"]
                     if os.path.exists(p)), None)
    if img_path is None:
        print(f"[WARN] 對應影像不存在 → {txt_path}")
        continue
    h_img, w_img = cv2.imread(img_path).shape[:2]

    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            x = (cx - w/2) * w_img
            y = (cy - h/2) * h_img
            w_px, h_px = w * w_img, h * h_img
            box = (x, y, w_px, h_px)
            if int(cls) == FEEDER_ID:
                feeders[img_name].append(box)
            elif int(cls) == CHICKEN_ID:
                chickens[img_name].append(box)

if all(len(f) == 0 for f in feeders.values()):
    raise RuntimeError("❌ 沒有任何 feeder 標註。")

def center(b): x,y,w,h=b; return np.array([x+w/2, y+h/2])
group_dists = [[] for _ in range(N_GROUPS)]

# ---------- 計算距離 ----------
for img_name in chickens:
    f_centers = np.array([center(b) for b in feeders[img_name]])
    if len(f_centers) == 0: continue
    # K-Means 依 y 分組
    if len(f_centers) >= N_GROUPS:
        ys = f_centers[:,1].reshape(-1,1)
        kmeans = KMeans(n_clusters=N_GROUPS, n_init="auto", random_state=0).fit(ys)
        order = np.argsort(kmeans.cluster_centers_.flatten())
        mapping = {order[0]:0, order[1]:1}
        f_groups = [mapping[l] for l in kmeans.labels_]
    else:
        f_groups = [0]*len(f_centers)

    for cbox in chickens[img_name]:
        cc = center(cbox)
        dists = np.linalg.norm(f_centers - cc, axis=1)
        idx = int(np.argmin(dists))
        group_dists[f_groups[idx]].append(dists[idx])

# ---------- IQR 去離群 & threshold ----------
thresholds = []
for g in group_dists:
    g = np.asarray(g)
    orig_data.append(g)
    if g.size == 0:
        clean_data.append([])
        thresholds.append(None)
        continue
    q1,q3 = np.percentile(g,[25,75]); iqr=q3-q1; upper=q3+1.5*iqr
    g_cln = g[g<=upper]
    clean_data.append(g_cln)
    thresholds.append(float(np.percentile(g_cln,90)))

print("\n=== 推論閾值 (px) ===")
for i,t in enumerate(thresholds):
    print(f"Group {i}: {t:.2f}" if t else f"Group {i}: 無樣本")

# ======================================================================
#                          繪　圖　區
# ======================================================================

# ① 原 / clean 箱形圖 ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8,4))
pos, lbl, pdata = [], [], []
for i,(r,c) in enumerate(zip(orig_data,clean_data)):
    pos += [i*3+1,i*3+2]
    lbl += [f"G{i}-raw",f"G{i}-clean"]
    pdata += [r,c]
ax.boxplot(pdata, positions=pos, widths=0.6, showfliers=True, patch_artist=True)
ax.set_xticks(pos); ax.set_xticklabels(lbl,rotation=30)
ax.set_ylabel("distance (px)")
ax.set_title("origin vs. outlier-removed")
plt.tight_layout(); plt.savefig("distance_boxplot.png", dpi=150)
print("[OK] distance_boxplot.png saved")

# ② 散點圖 --------------------------------------------------------------
rng = np.random.default_rng(0)
fig2, ax2 = plt.subplots(figsize=(6,4))
colors = ["tab:blue","tab:orange"]
for gid,g in enumerate(orig_data):
    if g.size==0: continue
    x = np.full_like(g,gid,dtype=float)+rng.uniform(-.12,.12,g.size)
    ax2.scatter(x,g,alpha=.6,s=18,color=colors[gid],
                label=f"Group {gid}" if f"Group {gid}" not in ax2.get_legend_handles_labels()[1] else "")
ax2.set_xticks([0,1]); ax2.set_xticklabels(["Group 0","Group 1"])
ax2.set_ylabel("distance (px)")
ax2.set_title("scatter distribution")
ax2.legend(frameon=False)
plt.tight_layout(); plt.savefig("distance_scatter.png", dpi=150)
print("[OK] distance_scatter.png saved")

# ③ 仿範例：箱形 + 常態密度 --------------------------------------------
def plot_groups_box_density(orig_data, clean_data, out_png="group_box_density.png"):
    n = len(orig_data); fig_h = 3*n
    fig, axes = plt.subplots(n,1,figsize=(6,fig_h),constrained_layout=True)
    if n==1: axes=[axes]
    for gid,(ax,raw,cln) in enumerate(zip(axes,orig_data,clean_data)):
        if len(cln)==0: ax.axis("off"); continue
        bp = ax.boxplot(cln,vert=True,positions=[0],widths=0.4,patch_artist=True,
                        showfliers=True,
                        boxprops=dict(facecolor="#FF9999",edgecolor="black"),
                        medianprops=dict(color="yellow",linewidth=2),
                        whiskerprops=dict(color="black"),capprops=dict(color="black"),
                        flierprops=dict(marker="o",markersize=4,markerfacecolor="green",
                                        markeredgecolor="green",alpha=0.7))
        q1,q3=np.percentile(cln,[25,75]); med=np.median(cln); iqr=q3-q1
        lower,upper=q1-1.5*iqr,q3+1.5*iqr
        for y,text in [(med,"Median"),(q1,"Q1"),(q3,"Q3"),
                       (upper,"Q3+1.5IQR"),(lower,"Q1-1.5IQR")]:
            ax.text(0.15,y,text,va="center",fontsize=8)
        kde = gaussian_kde(raw); xs=np.linspace(raw.min(),raw.max(),300); ys=kde(xs)
        ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); ax2.axis("off")
        ax2.plot(ys/ys.max()*0.4, xs, color="#1f77b4", linewidth=1)
        ax2.fill_betweenx(xs,0,ys/ys.max()*0.4,
                          where=(xs>=q1)&(xs<=q3),color="#FFBBBB",alpha=0.7)
        ax2.fill_betweenx(xs,0,ys/ys.max()*0.4,
                          where=(xs<q1)|(xs>q3),color="#AEC7E8",alpha=0.6)
        ax.set_title(f"Group {gid}",loc="left",pad=6)
        ax.set_ylabel("Distance (px)")
        ax.set_xticks([])
    fig.suptitle("Feeder–Chicken Distance\n(Box & Density)",fontsize=14)
    plt.savefig(out_png,dpi=150)
    print(f"[OK] {out_png} saved")

plot_groups_box_density(orig_data, clean_data)
