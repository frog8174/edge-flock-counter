"""Plot the full-day feeding rhythm (chickens around feeders, 5-min mean) for the README."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, parse_dates=["timestamp"])
# reindex to the 5-min grid so missing recordings show as gaps, not straight lines
d = d.set_index("timestamp").resample("5min").mean().reset_index()
SURF, INK, INK2, GRID, S1 = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
ax.plot(d.timestamp, d.chickens_around_mean, color=S1, lw=2)
pk = d.chickens_around_mean.idxmax()
ax.plot(d.timestamp[pk], d.chickens_around_mean[pk], "o", ms=7, color=S1, mec=SURF, mew=2)
ax.annotate(f"peak {d.chickens_around_mean[pk]:.0f} @ {d.timestamp[pk]:%H:%M}", (d.timestamp[pk], d.chickens_around_mean[pk]),
            xytext=(8, 4), textcoords="offset points", color=INK, fontsize=9)
ax.set_title("Chickens around feeders through one day (2025-05-14, 5-min mean, on-device inference)",
             loc="left", color=INK, fontsize=11)
ax.set_ylabel("chickens", color=INK2)
ax.set_ylim(bottom=0)
ax.text(0.995, 0.02, "gaps = no recording", transform=ax.transAxes, ha="right", color=INK2, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.grid(axis="y", color=GRID, lw=0.8)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
fig.tight_layout()
fig.savefig(out, facecolor=SURF)
print("[ok]", out)
