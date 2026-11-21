#!/usr/bin/env python3
"""
從 timeseries_analysis.py 的輸出，產生互動式 HTML 報告（雙主題、可 hover 看數值）。

用法：
  python3 generate_report.py \\
    --summary-dir results/20250514_summary \\
    --title 雞舍餵食節律 \\
    --date 2025-05-14 \\
    --out results/20250514_report.html

  # 有 daily_infer.py 產生的 tegrastats_batch.csv（僅 Jetson 平台才有）就會多畫溫度/功耗面板：
  python3 generate_report.py --summary-dir ... --tegrastats results/20250514/tegrastats_batch.csv --out ...

樣式與圖表渲染邏輯來自 dataviz skill 的規範（CVD-safe 雙色、雙主題 token、hover 十字準線 + tooltip、
資料缺口用灰底標示而非插值假造）。模板檔案在 templates/report_template.html，改樣式改那份，
這支腳本只負責算數字、填格子。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR.parent / "templates" / "report_template.html"


def build_chip(text, warn=False):
    cls = "chip warn" if warn else "chip"
    return f'      <span class="{cls}">{text}</span>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", required=True,
                    help="timeseries_analysis.py 的 --out 資料夾（要有 raw_timeseries.csv 和 timeseries_5min.csv）")
    ap.add_argument("--tegrastats", default=None,
                    help="daily_infer.py 產生的 tegrastats_batch.csv（選填，非 Jetson 平台通常沒有這個檔）")
    ap.add_argument("--title", default="雞舍餵食節律")
    ap.add_argument("--date", default=None, help="標示用日期字串，預設從資料自動抓")
    ap.add_argument("--model-note", default="TNC_v11x_0511（YOLO11x @960）",
                    help="footer 方法論註記要顯示的模型名稱")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    raw = pd.read_csv(summary_dir / "raw_timeseries.csv")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    bucket5 = pd.read_csv(summary_dir / "timeseries_5min.csv")
    bucket5["timestamp"] = pd.to_datetime(bucket5["timestamp"])

    date_str = args.date or raw["timestamp"].dt.date.iloc[0].isoformat()
    n_videos = raw["video_file"].nunique()
    n_frames = len(raw)
    t_min, t_max = raw["timestamp"].min(), raw["timestamp"].max()
    coverage = f"{t_min:%H:%M}–{t_max:%H:%M}"

    # peak / trough of around_rate on the 5-min bucket
    idx_peak = bucket5["around_rate_mean"].idxmax()
    idx_trough = bucket5["around_rate_mean"].idxmin()
    peak_time = bucket5.loc[idx_peak, "timestamp"].strftime("%H:%M")
    peak_val = bucket5.loc[idx_peak, "around_rate_mean"] * 100
    trough_time = bucket5.loc[idx_trough, "timestamp"].strftime("%H:%M")
    trough_val = bucket5.loc[idx_trough, "around_rate_mean"] * 100

    avg_rate = raw["around_rate"].mean() * 100
    avg_total = raw["chickens_total"].mean()
    max_total = raw["chickens_total"].max()

    # detect coverage gaps directly from the merged raw timeline (same rule as the chart)
    diffs = raw["timestamp"].diff().dt.total_seconds().dropna()
    big_gaps = diffs[diffs > 90]  # bigger than the inherent ~60s recording gap between clips
    gap_minutes = round(big_gaps.sum() / 60)
    has_gaps = len(big_gaps) > 0

    # ── meta chips ──
    chips = [
        build_chip(f"<b>{n_videos}</b> 支影片"),
        build_chip(f"<b>{n_frames:,}</b> 幀（{coverage} 覆蓋範圍內）"),
        build_chip(f"<b>{date_str}</b>"),
    ]
    if has_gaps:
        chips.append(build_chip(
            f"⚠ 偵測到 <b>{len(big_gaps)}</b> 段資料缺口，共 <b>{gap_minutes}</b> 分鐘，圖表中灰色區塊為缺口",
            warn=True))
    meta_chips = "\n".join(chips)

    lede = (f"{date_str} 全天雞舍影像，逐幀跑 {args.model_note} 偵測雞隻，"
           f"比對固定餵食器座標判斷「是否在餵食器周圍」。")

    # ── panel descriptions (auto-generated from the numbers, not hand-written per-dataset narrative) ──
    desc_total = (f"5 分鐘平均的畫面雞隻總數，範圍 {bucket5['chickens_total_mean'].min():.0f}–"
                 f"{bucket5['chickens_total_mean'].max():.0f} 隻，全天平均 {avg_total:.0f} 隻。"
                 f"這個數字反映的是鏡頭視野內可見的雞群分布，不是雞舍總雞數。")
    desc_rate = (f"「畫面中有多少比例的雞落在餵食器判定門檻內」。全天平均 {avg_rate:.1f}%，"
                 f"最高峰 {peak_time}（{peak_val:.1f}%），最低點 {trough_time}（{trough_val:.1f}%）。")
    desc_groups = "餵食器依畫面 y 座標分兩層（近鏡頭 group 1 / 遠鏡頭 group 0，補償透視變形），下圖比較兩層周圍的雞隻數隨時間變化。"

    # ── thermal panel (optional) ──
    thermal_records = []
    thermal_section = ""
    if args.tegrastats and Path(args.tegrastats).exists():
        tegra = pd.read_csv(args.tegrastats)
        tegra = tegra.dropna(subset=["temp_c"])
        if len(tegra):
            tegra["dt"] = pd.to_datetime(tegra["unix_ts"], unit="s")
            tegra["bucket"] = tegra["dt"].dt.floor("5min")
            g = tegra.groupby("bucket").agg(
                temp=("temp_c", "mean"), power=("power_mw", "mean"), gpu=("gpu_pct", "mean"))
            for ts, row in g.iterrows():
                thermal_records.append({
                    "t": ts.strftime("%H:%M"),
                    "temp": round(row["temp"], 1),
                    "power": round(row["power"] / 1000, 2) if pd.notna(row["power"]) else None,
                    "gpu": round(row["gpu"], 0) if pd.notna(row["gpu"]) else None,
                })
            run_minutes = (tegra["dt"].max() - tegra["dt"].min()).total_seconds() / 60
            t_start, t_peak, t_end = tegra["temp_c"].iloc[0], tegra["temp_c"].max(), tegra["temp_c"].iloc[-1]
            avg_power = tegra["power_mw"].mean() / 1000
            desc_temp = (f"連續 {run_minutes:.0f} 分鐘推論負載下的溫度與功耗。溫度從 {t_start:.0f}°C 到峰值 "
                        f"{t_peak:.0f}°C，結束時 {t_end:.0f}°C；平均功耗 {avg_power:.1f} W。")
            thermal_section = f'''  <section class="panel">
    <div class="panel-head">
      <h2>硬體穩態：溫度 &amp; 功耗</h2>
      <span class="note">連續 {run_minutes:.0f} 分鐘推論負載</span>
    </div>
    <p class="panel-desc">{desc_temp}</p>
    <div class="chart-wrap"><svg class="chart" id="chart-temp"></svg></div>
  </section>'''

    # ── footer notes ──
    footer_items = [
        f"推論：<code>{args.model_note}</code>，關閉 TTA",
        "餵食器判定沿用 <code>config/cached_feeders.json</code> 座標與距離門檻，未逐幀重新偵測餵食器（相機固定；相機一旦移位需重新標定座標）",
        "TensorRT engine 若有使用：屬於本機編譯產物，換 GPU / 換 TensorRT 版本需要重新匯出，不可直接搬用",
        "原始逐幀 / 1 分鐘 / 5 分鐘三種粒度 CSV 都在同一個 summary 資料夾，供之後重新分桶或串接其他分析",
    ]
    if has_gaps:
        footer_items.insert(1, "灰色區塊代表該時段沒有影片資料，非「量到零」——避免誤讀為活動中斷")
    footer_notes = "\n".join(f"      <li>{item}</li>" for item in footer_items)

    # ── chart data (5-min bucket) ──
    chart_records = []
    for _, r in bucket5.iterrows():
        chart_records.append({
            "t": r["timestamp"].strftime("%H:%M"),
            "total": round(r["chickens_total_mean"], 1),
            "around": round(r["chickens_around_mean"], 1),
            "rate": round(r["around_rate_mean"] * 100, 2),
            "g0": round(r["group0_far_mean"], 1),
            "g1": round(r["group1_near_mean"], 1),
        })

    tpl = TEMPLATE_PATH.read_text()
    subs = {
        "%%TITLE%%": args.title,
        "%%EYEBROW%%": "邊緣推論時序分析",
        "%%LEDE%%": lede,
        "%%META_CHIPS%%": meta_chips,
        "%%STAT_PEAK_TIME%%": peak_time,
        "%%STAT_PEAK_VAL%%": f"{peak_val:.1f}%",
        "%%STAT_TROUGH_TIME%%": trough_time,
        "%%STAT_TROUGH_VAL%%": f"{trough_val:.1f}%",
        "%%STAT_AVG_RATE%%": f"{avg_rate:.1f}%",
        "%%STAT_FRAME_COUNT%%": f"{n_frames:,}",
        "%%STAT_AVG_TOTAL%%": f"{avg_total:.0f}",
        "%%STAT_MAX_TOTAL%%": f"{max_total:.0f}",
        "%%DESC_TOTAL%%": desc_total,
        "%%DESC_RATE%%": desc_rate,
        "%%DESC_GROUPS%%": desc_groups,
        "%%THERMAL_SECTION%%": thermal_section,
        "%%FOOTER_NOTES%%": footer_notes,
        "%%CHART_DATA_JSON%%": json.dumps(chart_records),
        "%%THERMAL_DATA_JSON%%": json.dumps(thermal_records),
    }
    html = tpl
    for k, v in subs.items():
        html = html.replace(k, v)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"報告已產生 → {out_path}")
    print(f"  {n_videos} 支影片, {n_frames:,} 幀, 覆蓋 {coverage}"
         + (f", 缺口 {gap_minutes} 分鐘" if has_gaps else ", 無缺口"))


if __name__ == "__main__":
    main()
