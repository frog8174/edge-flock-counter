#!/usr/bin/env python3
"""
把 daily_infer.py 產生的每支影片逐幀 CSV，合併成一天的時序分析。

輸出三種粒度（都做）：
  - raw      ：逐幀（5fps 原始解析度），供之後自行重新分桶
  - 1min     ：每分鐘平均，看進食尖峰與節律
  - 5min     ：每 5 分鐘平均，看大趨勢

用法：
  python3 timeseries_analysis.py --dir results/20250514 \
      --out results/20250514_summary
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="daily_infer.py 輸出的逐幀 CSV 資料夾")
    ap.add_argument("--out", required=True, help="彙整結果輸出資料夾")
    args = ap.parse_args()

    src_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(f for f in src_dir.glob("*.csv") if f.name != "tegrastats_batch.csv")
    if not csvs:
        raise SystemExit(f"{src_dir} 底下沒有找到逐幀 CSV")

    dfs = [pd.read_csv(f) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")  # mixed with/without fractional seconds
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_videos = len(csvs)
    n_frames = len(df)
    span = f"{df['timestamp'].min():%H:%M} → {df['timestamp'].max():%H:%M}"
    print(f"合併 {n_videos} 支影片、{n_frames} 幀，涵蓋 {span}")

    # ── raw ──
    raw_path = out_dir / "raw_timeseries.csv"
    df.to_csv(raw_path, index=False)

    # ── 分桶彙整 ──
    def bucket(freq, label):
        # pandas 1.3.5 的 Resampler.agg() 不支援關鍵字式 named aggregation，
        # 改用 dict 形式 + 事後改欄名
        g = df.set_index("timestamp").resample(freq).agg({
            "chickens_total": ["count", "mean"],
            "chickens_around_feeder": "mean",
            "around_rate": "mean",
            "around_group0_far": "mean",
            "around_group1_near": "mean",
        })
        g.columns = ["frames", "chickens_total_mean", "chickens_around_mean",
                     "around_rate_mean", "group0_far_mean", "group1_near_mean"]
        # frames==0 的桶代表這段時間根本沒有影片覆蓋（錄影間隔或影片缺檔），
        # 不是「量到 0」，必須整列丟掉，不能只靠 dropna(how="all")
        # （count 聚合對空桶回傳 0 而非 NaN，其他欄位才是 NaN，兩者不同調）
        g = g[g["frames"] > 0]
        p = out_dir / f"timeseries_{label}.csv"
        g.round(3).to_csv(p)
        print(f"  {label:>6} 分桶 → {p}  ({len(g)} 個時間點)")
        return g

    b1 = bucket("1min", "1min")
    b5 = bucket("5min", "5min")

    # ── 找進食高峰（用 1 分鐘分桶的 around_rate）──
    peak = b1["around_rate_mean"].idxmax()
    peak_val = b1["around_rate_mean"].max()
    trough = b1["around_rate_mean"].idxmin()
    print(f"\n進食比例最高峰：{peak:%H:%M}（around_rate={peak_val:.1%}）")
    print(f"進食比例最低點：{trough:%H:%M}（around_rate={b1['around_rate_mean'].min():.1%}）")
    print(f"全天平均在餵食器周圍比例：{df['around_rate'].mean():.1%}")
    print(f"全天平均雞隻總數/幀：{df['chickens_total'].mean():.0f}（max {df['chickens_total'].max()}）")

    print(f"\n原始逐幀 → {raw_path}")


if __name__ == "__main__":
    main()
