#!/usr/bin/env python3
"""GPU 單獨 vs GPU+DLA 並行 — 閒置的 DLA 到底能不能撿到吞吐？"""
import re, statistics, subprocess, threading, time
from pathlib import Path
import cv2
from trt_runner import TRTRunner

VIDS = sorted(Path("data/videos/20250501").glob("*.mp4"))

class W(threading.Thread):
    def __init__(s, eng, vid, secs):
        super().__init__(); s.eng, s.vid, s.secs = eng, vid, secs; s.n = 0; s.err=None
    def run(s):
        try:
            r = TRTRunner(s.eng); cap = cv2.VideoCapture(str(s.vid))
            ok, f = cap.read()
            for _ in range(2):
                if ok: r.infer(f)
            end = time.time() + s.secs
            while time.time() < end:
                ok, f = cap.read()
                if not ok: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                r.infer(f); s.n += 1
            cap.release()
        except Exception as e: s.err = str(e)

def power(secs):
    p = subprocess.Popen(["tegrastats","--interval","200"], stdout=subprocess.PIPE, text=True)
    w=[]; end=time.time()+secs
    while time.time()<end:
        l=p.stdout.readline()
        if m:=re.search(r"VDD_IN (\d+)mW", l or ""): w.append(int(m.group(1)))
    p.terminate(); return statistics.mean(w)/1000 if w else 0

def run(label, engines, secs=25):
    ws=[W(e, VIDS[i], secs) for i,e in enumerate(engines)]
    res={}
    t=threading.Thread(target=lambda: res.update(w=power(secs+2))); t.start()
    time.sleep(1)
    t0=time.time()
    for w in ws: w.start()
    for w in ws: w.join()
    wall=time.time()-t0; t.join()
    if any(w.err for w in ws):
        print(f"  {label}: ✗ {[w.err for w in ws if w.err][0][:120]}"); return None
    tot=sum(w.n for w in ws)
    print(f"  {label:<22} 聚合 {tot/wall:>6.2f} FPS   "
          + "  ".join(f"[{Path(w.eng).stem[:12]}] {w.n/wall:.2f}" for w in ws)
          + f"   {res.get('w',0):.1f} W")
    return tot/wall

print("每組跑 25 秒\n")
g = run("GPU 單獨",        ["models/TNC_v11x_0511.engine"])
d = run("DLA0 單獨",       ["models/TNC_dla0.engine"])
c = run("GPU + DLA0 並行", ["models/TNC_v11x_0511.engine", "models/TNC_dla0.engine"])
if g and c:
    print(f"\n  併用 DLA 的淨增益: {(c-g)/g*100:+.1f}%  ({g:.2f} → {c:.2f} FPS)")
