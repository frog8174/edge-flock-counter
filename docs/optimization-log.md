# Optimization log: real-time chicken counting on Jetson Orin NX

> **Core question:** how many poultry-house cameras can one Jetson Orin NX process in real time, replacing Raspberry Pis that only record?
>
> This is not a synthetic benchmark. The original study's future work says *"increase more systems to cover more area"*: a house is 113 m × 14.8 m, and there are 10 of them. Multi-stream inference is the project's real next step.

Device: NVIDIA Jetson Orin NX 16 GB Engineering Reference Developer Kit
Software: JetPack R36.4.4 (L4T) · Ubuntu 22.04 · CUDA 12.6 · TensorRT 10.3 · cuDNN 9.3

Each phase records **what changed, how the numbers moved, and why**. Experiments that failed are kept on purpose.
中文版：[optimization-log.zh-TW.md](optimization-log.zh-TW.md)

---

## Context

### Before: record on site, infer offline

```
Site:    Raspberry Pi 4B + 130° camera (1920×1080 @ 5 fps), recording only
           ↓ videos copied by hand to the lab NAS
Offline: YOLO11n @640    feeders          (AP 97.5%)
         1-D K-means on feeder y → near / far row (perspective)
         YOLO11x @960    chickens, 650–750 per frame (AP 91.6%)
         distance threshold [45.10, 94.23] px → "around feeder"
Result:  P 61.6%  R 61.7%  F1 0.617
```

On site, the Pi is only a recorder and all the intelligence is offline. Moving inference to an on-site Jetson turns "record, then analyze" into **live sensing**.

### Key edge fact: the real-time bar is 5 FPS, not 30 FPS

The field camera runs at 5 FPS. That bar is 6× lower than 30 FPS, so the number of streams one device can carry goes up by the same factor.

### Data and workload

| | |
|---|---|
| Video | 66 clips, 1920×1080 H.264 @ 5 fps, 300 s each, 07:00 → 19:54, about 99k frames |
| Chicken model | YOLO11x, 56.9M params, imgsz 960, single class |
| Cost per frame | about 438 GFLOPs (194.9 at 640 × 2.25); with TTA (`augment=True`) **about 1300 GFLOPs** |
| Feeder model | not needed at runtime. The camera is fixed, so feeder boxes are computed once and cached (`pipeline/config/cached_feeders.json`). That saves a whole model per frame. |

---

## Phase 0: baseline, and unlocking the clocks

### 0-A. Methodology check with YOLO11n

| Backend | ms/frame | FPS |
|---|---:|---:|
| PyTorch | 33.38 | 30.0 |
| TensorRT FP16 | 28.55 | 35.0 |
| Speedup | | **only 1.17×** |

Power went from 7.82 W idle to 8.17 W under load, an increase of only 0.35 W.

**Why only 1.17×?**
1. **The GPU sat at 306 MHz the whole time (max 918 MHz).** Short batch-1 bursts never trigger DVFS to raise the clock. The 0.35 W power increase confirms the GPU was barely working.
2. **The bottleneck was Python, not GPU kernels.** TensorRT only accelerates the forward pass. Letterboxing and NMS run on one CPU core. With a 2.6M-parameter model, the GPU finishes and then waits.
3. **Batch 1** wastes parallelism, so kernel-launch overhead dominates.

**This baseline measured pipeline efficiency, not hardware capability.**

### 0-B. Lock the clocks

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

| Clock | Before | After |
|---|---:|---:|
| GPU | 306 MHz | **918 MHz** (3.0×) |
| CPU0 | 1344 MHz | **1984 MHz** |
| EMC | n/a | **3199 MHz** |

With DVFS off, the numbers become reproducible. **Every number below was measured with the clocks locked.**

---

## Phase 1: TensorRT for the real model

ONNX export gives 218 MB, with input 1×3×960×960 and output 1×5×18900. The FP16 engine takes **1072 s** to build and is 112 MB.

| Config | FPS | Mean latency | p99 | Power | Real-time headroom (5 fps) |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 + TTA (thesis) | 2.82 | 349.1 ms | 362.2 | 19.7 W | 0.56× ❌ |
| PyTorch FP32, TTA off | 6.21 | 155.8 ms | 163.0 | 19.3 W | 1.24× ✅ |
| **TensorRT FP16** | **11.03** | **85.6 ms** | 96.8 | 17.8 W | **2.21×** ✅ |

Turning TTA off gives 2.20×, and TensorRT then gives another 1.78×, for **3.91× in total**.

> ⚠️ **Trap:** with `augment=True` on a TensorRT engine, the result is identical (11.04 vs 11.03 FPS) because **TTA is silently ignored**. The engine input shape is fixed, and TTA needs multiple scales. Moving to TensorRT means giving up TTA: it is a trade-off, not a free speedup.

---

## Phase 2: multi-stream scalability

| Streams | Aggregate FPS | Slowest stream | Mean latency | p99 | GPU | Power | FPS/W | Real-time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 11.03 | 11.03 | 85.6 ms | 96.8 | 52% | 17.8 W | 0.62 | 2.21× ✅ |
| **2** | **12.09** | **6.04** | 159.9 ms | 223.8 | 82% | 21.8 W | 0.56 | **1.21×** ✅ |
| 3 | 12.19 | 4.01 | 239.7 ms | 314.4 | 85% | 22.7 W | 0.54 | 0.80× ❌ |
| 4 | 12.25 | 2.99 | 319.3 ms | 459.9 | 88% | 23.3 W | 0.53 | 0.60× ❌ |

**Real-time capacity: 2 streams per device.**

From the second stream on, aggregate throughput stays at 12.1–12.3 FPS, while latency climbs linearly from 86 ms to 319 ms. **A throughput plateau with linearly growing latency is the textbook signature of saturation.** Scaling efficiency is 12.09 / (11.03 × 2) = 54.8%.

> **Headroom is not capacity.** 2.21× headroom for one stream suggests two streams fit with room to spare. In practice there is only 1.21× left, because one stream already uses 52% of the GPU.

---

## Phase 3: use the idle hardware?

### 3-A. Measure first, then prioritize

| Frame-time component | ms | Share |
|---|---:|---:|
| GPU inference | 85.6 | 94.1% |
| CPU decode (OpenCV) | 5.0 | 5.9% |

NVDEC hardware decode can save **at most 5.9%** of frame time. The priority order therefore becomes INT8 > DLA > NVDEC.

### 3-B. DLA offload ❌

TensorRT `can_run_on_DLA`: **648 of 694 layers (93.4%)** can run on the DLA. The unsupported ones are CONSTANT, SHUFFLE, MATRIX_MULTIPLY (C2PSA attention), SLICE and CAST.

| Device | Latency | Throughput |
|---|---:|---:|
| GPU FP16 | 85.6 ms | 11.7 FPS |
| **DLA0 FP16** (GPU fallback) | **909 ms** | **1.1 FPS** |

The DLA is **10.6× slower**. All 11 fallback layers are in the detection head (`model.23`), at the end of the network. That means there is one DLA→GPU hand-off and no ping-ponging, so 909 ms is the DLA's real compute time. There is also a hard limit: the head's 18900-anchor output exceeds the DLA's maximum dimension of 8192.

> The Orin NX DLA is built for efficiency on small models. A 57M-parameter YOLO11x at 960² is outside that range.

### 3-C. Native TensorRT runner

INT8 and DLA engines carry no Ultralytics metadata, so `YOLO()` cannot load them. To run them, I wrote [`benchmarks/trt_runner.py`](../benchmarks/trt_runner.py), which calls the TensorRT API directly and runs NMS on the GPU with torchvision.

| Path | FPS | Latency |
|---|---:|---:|
| Ultralytics + engine | 11.03 | 85.6 ms |
| **Native runner, same engine** | **11.52** | **81.6 ms** |

That is **+4.4%** with matching detections (229 vs 228). From here on, every engine runs through this path so the comparisons are fair.

### 3-D. INT8 (PTQ) ❌ accuracy collapse

Calibration used 210 real frames across seven time slots from 07:00 to 17:18, covering the light changes through the day. The engine is 61 MB, and calibration plus build took about 26 minutes.

| | FPS | Latency | Power | FPS/W | Detections |
|---|---:|---:|---:|---:|---:|
| FP16 | 11.52 | 81.6 ms | 18.2 W | 0.63 | 229 |
| INT8 | **18.69** | **48.3 ms** | 15.5 W | **1.21** | **82** ⚠️ |

That is 1.62× faster and 1.92× more efficient, **but it misses 64% of the chickens.** Against FP16 as the reference (IoU ≥ 0.5, 25 frames), INT8 gets recall 36.7%, precision 91.0% and F1 0.523.

High precision with low recall suggests the confidence scores are compressed. Sweeping the threshold shows that this is not the cause:

| INT8 conf | Detections | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|
| 0.49 | 116 | 40.8% | 90.2% | **0.562** |
| 0.25 | 159 | 46.0% | 69.8% | 0.554 |
| 0.10 | 299 | 51.7% | 51.1% | 0.514 |
| 0.05 | 346 | 53.6% | 39.8% | 0.457 |

At conf 0.05, INT8 returns more boxes than FP16 does, and recall is still only 53.6%. **The features themselves are gone. INT8 is detecting different things.**

### 3-E. Mixed precision ❌

Hypothesis: the damage comes from the C2PSA attention block (softmax and matmul have a wide dynamic range) and from the detection head. To test it, 138 layers (19.9%) were forced to FP16.

| | Detections | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| INT8 | 98 | 36.7% | 91.0% | 0.523 |
| Mixed | 91 | **34.4%** | 90.4% | 0.498 |

There was no improvement, so the hypothesis is wrong.

Gotcha: CONSTANT, GATHER and CAST layers can hold Int64 weights. Forcing them to FP16 fails with `cannot use precision Half with weights of type Int64`, so they must be excluded.

### 3-F. The real cause: object size

Recall bucketed by the area of the FP16 box:

| Size | Count | INT8 recall |
|---|---:|---:|
| Tiny, < 900 px² | 882 | **10.7%** |
| Small, 900–2500 | 2605 | 32.2% |
| Medium, 2500–6400 | 1205 | 71.5% |
| Large, > 6400 | 141 | **82.3%** |

This one finding explains all three earlier results:
1. **Mixed precision didn't help** because the damage is in the early and middle backbone feature maps that carry fine texture, not in the attention layers.
2. **Lowering the threshold didn't help** because the features are gone, not just suppressed.
3. **This scene is inherently hostile to PTQ.** It is full of tiny, dense, low-contrast targets (white birds on pale litter), exactly the case 256 quantization levels can't represent.

Next options: QAT instead of PTQ, or a tighter camera framing so the targets are larger.

### 3-G. GPU + DLA concurrently ❌

| Config | Aggregate FPS | Power |
|---|---:|---:|
| GPU alone | 10.87 | 22.0 W |
| DLA0 alone | 0.99 | 14.9 W |
| **GPU + DLA0** | **8.95** | 24.0 W |

**Net −17.7%.** The GPU drops from 10.87 to 8.02 FPS, and the DLA adds only 0.93. The DLA engine's head still falls back to the GPU, both share LPDDR5 bandwidth, and power approaches the cap. **The DLA takes more from the GPU than it contributes.**

### Phase 3 summary

| Approach | Result | Kept? |
|---|---|---|
| Native TensorRT runner | +4.4% | ✅ |
| DLA offload | 10.6× slower; −17.7% alongside the GPU | ❌ |
| INT8 PTQ | 1.62× faster, recall 36.7% | ❌ |
| Mixed-precision INT8 | recall 34.4% | ❌ |
| NVDEC decode | targets only 5.9% of frame time | 🟡 deprioritized |

**Adopted: TensorRT FP16 + native runner.** The idle hardware turned out to be idle for good reasons. **The point of measuring is to make decisions, not to produce flattering numbers.**

---

## Unified sweep

All six configurations were re-run under identical conditions with [`benchmarks/optimization_matrix.py`](../benchmarks/optimization_matrix.py). Each one is compared against config A's detections. Results: [`results/optimization_matrix.json`](../benchmarks/results/optimization_matrix.json). This sweep is the table shown in the README. Absolute FPS differs from the phase tables above by about 3% between runs.

---

## Full-day production run

[`pipeline/scripts/daily_infer.py`](../pipeline/scripts/daily_infer.py) processed 86 clips (2025-05-14, 07:00 → 19:54, about 129k frames) **on the device in 202.8 minutes at 10.9 FPS**, with `tegrastats` sampled throughout. The time series and the HTML report come from `timeseries_analysis.py` and `generate_report.py`.

---

## Field scaling

Layout: 113 m houses, one camera every 8 m → 15 cameras per house × 10 houses = **150 cameras**.

| | Today (central, offline) | Edge |
|---|---|---|
| On-site devices | 150 RPi (record only) | **75 Jetson** (live inference) |
| Data moved per day | 1755 GB | 1.40 GB |
| 12-week flock cycle | **147 TB** | 0.12 TB |
| Reduction | n/a | **1250×** |
| Result latency | after copying and batch processing | immediate |
| On-site power | 450 W | 1632 W |

To keep up with all cameras, the central option needs a sustained 406 FPS (35M frames per day). Even at the optimized rate, that is about 34 equivalent machines, and 147 TB still has to be moved first.

> **Read the power line honestly.** Edge uses 3.6× more power on site, but the Pis' 450 W only pays for recording. The offline machines, NAS storage and transport of 147 TB are not in that number.

---

## Honest boundaries

- **Detections per frame:** 228–321 here, versus 650–750 labeled in the training set. The likely causes are the inherited `conf=0.49` and missed far, small birds. NMS load is therefore a little below reality, which does not change the saturation conclusion.
- **Accuracy after FP16:** measured only as agreement with the thesis setup's detections (recall 0.87, precision 0.97 vs TTA). Thesis-level P/R/F1 against ground truth has not been re-run.
- **Thermals:** measured over minutes (peak 70.2 °C). Throttling over weeks in a hot, dusty house remains untested.
- **Feeder calibration is static.** It uses the 11 cached feeder positions, which were verified against the frames. If the camera moves, they must be recalibrated.

## Not yet done

| Idea | Why it might help |
|---|---|
| QAT (quantization-aware training) | The only realistic path to usable INT8 for tiny targets |
| DLA with a small model (YOLO11n @640) | Tests whether the DLA pays off within its intended range |
| GPU preprocessing + EfficientNMS plugin | Removes the remaining Python/CPU work at 650+ targets |
| Zero-copy NVMM buffers + `nvv4l2decoder` | Jetson's unified memory makes cudaMemcpy avoidable |
| DeepStream multi-stream | Production-grade batching across cameras |

## Environment prep (side quest)

The device's disk was 79% full. Clearing 116 GB brought it to 27%:
- `/var/log/uvcdynctrl-udev.log` had reached **29 GB**. This is a known `uvcdynctrl` bug where udev events append the same banner forever. I truncated it with `truncate -s 0`, not `rm`, so an open file handle couldn't keep the space allocated, and added a logrotate rule. The rule needs `su root syslog`, because `/var/log` is `root:syslog 775`.
- 87 GB of stale Triton images and containers were removed. `docker images` without sudo **silently returns nothing** when the user isn't in the docker group, which briefly made it look as if there were no images.
