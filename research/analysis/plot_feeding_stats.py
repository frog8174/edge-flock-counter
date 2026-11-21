#!/usr/bin/env python3
# plot_feeding_stats.py
# ------------------------------------------------------------
# 讀取 feeding_stats.csv → 解析時間 → 處理缺格 → 繪圖輸出
# 產出：
#   1. eat_ratio 曲線 (含移動平均)
#   2. eat_count 與 total_chicks 疊合柱狀圖
#   3. 每小時平均 eat_ratio 盒鬍圖
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------- 使用者設定 ---------------------------------------------------
CSV_PATH          = Path("./runs/feeding_detection/20250507_fullday/feeding_stats.csv")
TIMEZONE          = "Asia/Taipei"   # 依實際時區調整
RESAMPLE_FREQ     = "1T"            # 缺格補齊採樣間隔：1T=1分鐘, 30S=30秒…
INTERPOLATE_LIMIT = 3               # 連續缺值 ≤3 個採樣點才內插，超過保持 NaN
ROLLING_WINDOW    = 5               # 移動平均視窗 (單位=採樣點)
SHOW_PLOTS        = True            # True=以視窗打開；False=只存檔
OUT_DIR           = Path("./analysis_plots")
# -------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------- 1. 讀檔與絕對時間戳 ---------------------------------------
df = pd.read_csv(CSV_PATH)
def parse_start(video_name: str) -> pd.Timestamp:
    # e.g. 20250507_1600.mp4 → 2025-05-07 16:00:00
    stem = Path(video_name).stem
    date_part, time_part, *_ = stem.split("_")
    return pd.to_datetime(date_part + time_part, format="%Y%m%d%H%M").tz_localize(TIMEZONE)

df["start_dt"]     = df["video"].apply(parse_start)
df["timestamp_dt"] = df["start_dt"] + pd.to_timedelta(df["timestamp_sec"], unit="s")
df.sort_values("timestamp_dt", inplace=True)
df.set_index("timestamp_dt", inplace=True)

# ------------- 2. 缺格修補（重採樣 → 插值 / 前向補）-----------------------
# a) 先對 eat_ratio 進行線性插值（短缺格才補）
eat_ratio = (
    df["eat_ratio"]
    .resample(RESAMPLE_FREQ)
    .mean()                       # 聚合採樣區間內同時段多筆值
    .interpolate("linear", limit=INTERPOLATE_LIMIT)
)

# b) eat_count / total_chicks 用前向填補 (缺值處保持上一次檢測結果)
eat_count = (
    df["eat_count"]
    .resample(RESAMPLE_FREQ)
    .max()
    .fillna(method="ffill")
)
total_ch  = (
    df["total_chicks"]
    .resample(RESAMPLE_FREQ)
    .max()
    .fillna(method="ffill")
)

# ------------- 3. 繪圖 ---------------------------------------------------
# 圖 1：進食比例曲線 + 移動平均
plt.figure(figsize=(12, 4))
plt.plot(eat_ratio.index, eat_ratio, label="Eat Ratio (raw)", alpha=0.4)
plt.plot(eat_ratio.rolling(ROLLING_WINDOW, min_periods=1).mean(),
        label=f"{ROLLING_WINDOW}-pt Moving Avg", linewidth=2)
plt.ylabel("Eat Ratio")
plt.xlabel("Time")
plt.title("Feeding Ratio Over Time")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "eat_ratio_curve.png", dpi=300)
if SHOW_PLOTS: plt.show()

# 圖 2：同一張圖疊合柱狀 – eat_count 與 total_chicks
plt.figure(figsize=(12, 4))
plt.bar(eat_count.index, eat_count, width=0.03, label="Eat Count",
        align="center", alpha=0.6)
plt.bar(total_ch.index, total_ch, width=0.03, label="Total Chickens",
        align="center", alpha=0.3)
plt.ylabel("Count")
plt.xlabel("Time")
plt.title("Chicken Counts Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "count_bar.png", dpi=300)
if SHOW_PLOTS: plt.show()

# 圖 3：每小時平均 eat_ratio 盒鬍圖，觀察時段分布
hourly = eat_ratio.resample("1H").mean()
plt.figure(figsize=(6, 4))
hourly = hourly.dropna()
plt.boxplot(hourly, vert=True, labels=["hourly\nmean"])
plt.ylabel("Eat Ratio")
plt.title("Hourly Mean Eat Ratio (Distribution)")
plt.tight_layout()
plt.savefig(OUT_DIR / "hourly_boxplot.png", dpi=300)
if SHOW_PLOTS: plt.show()

print(f"\n✅  圖片已輸出到：{OUT_DIR.resolve()}")
