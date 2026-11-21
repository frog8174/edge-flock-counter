# Jetson Orin NX 邊緣運算優化日誌 — 雞舍即時計數

> **核心問題**：一台 Jetson Orin NX 能同時即時處理幾路雞舍攝影機，
> 取代幾台只會錄影的 Raspberry Pi？
>
> 這不是合成 benchmark。原研究的 Future Work 明訂要
> *"Increase more system to cover more area"*（雞舍 113m × 14.8m × 10 排），
> 多路串流正是這個專案真正的下一步。

裝置：NVIDIA Jetson Orin NX Engineering Reference Developer Kit
JetPack：R36.4.4 (L4T) / Ubuntu 22.04 / CUDA 12.6 / TensorRT 10.3 / cuDNN 9.3

---

## 專案脈絡

台大生機系研究：**Automatic Counting of Chickens Around Feeders Using CNN**
（Zi-Yuan Chen，指導教授 Dr. Yan-Fu Kuo，2025-05-20）

### 現況架構（優化前）

```
現場: Raspberry Pi 4B + 130° 相機 (1920×1080 @ 5fps)  ← 只負責錄影，不做推論
  ↓  影片人工搬到實驗室 NAS
離線: YOLO11n @640  抓餵食器  (AP 97.5%)
      1D K-Means 依 y 座標分兩層 ← 補償透視變形
      YOLO11x @960  抓雞 (650~750 隻/幀, AP 91.6%)
      距離門檻 [45.10, 94.23] px 判定「在餵食器旁」
結果: P=61.6%  R=61.7%  F1=0.617
```

**這就是 edge computing 的切入點**：目前 RPi 只是台錄影機，所有智慧都在離線端。
把推論搬到現場的 Jetson，就從「錄影 + 事後分析」變成「即時感測」。

### 關鍵 edge 特性：即時門檻是 5 fps，不是 30 fps

現場相機就是 5 fps。這是本專案最重要的 edge 洞察 ——
即時門檻低了 6 倍，可承載的串流數就直接放大。

### 資料

| 項目 | 規格 |
|---|---|
| 位置 | `~/Videos/20250501/` |
| 數量 | 66 支 mp4，4.7 GB |
| 規格 | 1920×1080 H.264 **@ 5 fps** |
| 每支 | 300 秒 / 1502 幀 |
| 時間覆蓋 | 0700 → 1954（完整日間餵食期） |
| 總計 | 6.6 小時 ≈ **99,132 幀** |

### 模型

| 檔案 | 說明 |
|---|---|
| `~/models/yolo11_pt/TNC_v11x_0511.pt` | 雞隻偵測，YOLO11x，56.9M params，imgsz=960 |
| `~/models/TNC.plan` | 舊 TensorRT engine，FP32 動態尺寸，TRT 10.3 可載入 |
| `~/models/TNC.engine` | ❌ 序列化版本不符，無法載入 |
| 餵食器模型 | ⚠️ 找不到，但 `cached_feeders.json` 已有算好的座標 |

> **重要洞察**：相機固定不動 → 餵食器位置是靜態的，只需算一次。
> 餵食器偵測本來就不該每幀跑。這是免費的算力節省。

### 單幀運算量

YOLO11x @960 ≈ **438 GFLOPs**（YOLO11x @640 是 194.9 GFLOPs，×(960/640)² = 2.25）
原程式 `augment=True` 開了 TTA → **約 1300 GFLOPs/幀**
再加上 650~750 個目標、`max_det=1000` 的 NMS

這是個夠重的 workload，不會像 YOLO11n 那樣測不出東西。

---

## Phase −1：環境整備 ✅ 完成（2026-08-19）

### 磁碟清理：174G → 58G，回收 116 GB

| 項目 | 回收 | 說明 |
|---|---|---|
| `/var/log/uvcdynctrl-udev.log` | 29 GB | Ubuntu `uvcdynctrl` 已知 bug，udev 事件觸發無限寫入 |
| Triton 映像檔 ×7 + 容器 ×9 | ~87 GB | 22.12→24.09 跨 7 版本，容器停止 13 個月 |

- `truncate -s 0`（非 `rm`，避免 file handle 未釋放導致空間不回收）
- 新裝 `logrotate` + `/etc/logrotate.d/uvcdynctrl`（10M 輪替）防止復發
  - ⚠️ 設定必須含 `su root syslog`，否則因 `/var/log` 是 `root:syslog drwxrwxr-x` 而拒絕輪替
- 保留 `nvidia/cuda` base image

> 踩坑：`docker images` 不加 sudo 會**靜默回空**，一度誤判「沒有映像檔」。
> 註：`~/models/TNC.plan`、`~/trt-repository/`、`~/triton/` 的模型檔都還在，
> 只有 Triton **server 映像檔**被移除，需要時可重新 pull。

---

## Phase 0：Baseline 與時脈解鎖 ✅ 完成（2026-08-19）

### 0-A. 先用 YOLO11n 建立方法論（非本專案模型）

| Backend | ms/frame | FPS |
|---|---|---|
| PyTorch | 33.38 | 30.0 |
| TensorRT FP16 | 28.55 | 35.0 |
| 加速比 | | **僅 1.17×** |

功耗：閒置 7.82 W → 推論 8.17 W（**增量僅 0.35 W**）

### 🔍 診斷：為什麼只有 1.17×？

**這段是整個專案最有價值的發現。面試時講這個，比報漂亮數字更有說服力。**

1. **GPU 全程卡在 306 MHz 怠速**（上限 918 MHz）
   短促、batch=1 的 workload 不足以讓 DVFS 調頻爬升 → 量到的是「GPU 沒醒來」的數字。
   佐證：推論功耗增量只有 0.35 W，GPU 根本沒出力。

2. **瓶頸不在 GPU kernel，在 Python 前後處理**
   TensorRT 只加速模型 forward pass。影像 resize/letterbox、NMS 全在 Python/CPU 單核跑。
   模型越小，這段佔比越高。YOLO11n 只有 2.6M 參數，GPU 早算完在等 CPU。

3. **batch=1 浪費 GPU 並行度**，kernel launch 開銷佔比極高。

**結論：baseline 反映的是 pipeline 效率，不是硬體能力。**

### 0-B. 解除時脈限制 ✅

```bash
sudo nvpmodel -m 0     # MAXN
sudo jetson_clocks     # 鎖定最高時脈，關閉 DVFS
```

| 時脈 | 前 | 後 | 倍數 |
|---|---|---|---|
| GPU | 306 MHz | **918 MHz** | 3.0× |
| CPU0 | 1344 MHz | **1984 MHz** | 1.48× |
| EMC (記憶體) | — | **3199 MHz** | max |

Active GPU TPCs: 4

> 副作用（正面）：關閉 DVFS 後 benchmark 數字才穩定可重現。
> DVFS 是量測雜訊的主要來源。**此後所有數據都在鎖頻狀態下取得。**

---

## Phase 1：雞隻模型 TensorRT 化 ✅ 完成

- [x] 確認 `TNC_v11x_0511.pt` 可載入（56.9M params, imgsz=960, class TNC）
- [x] ONNX 匯出成功（218 MB，輸入 1×3×960×960 → 輸出 1×5×18900）
- [x] FP16 engine 建置完成（**1072 秒**，112 MB）
- [x] 單路 baseline 完成
- [x] TTA 成本量測完成

### 單路優化鏈（每階段獨立量測）

| 設定 | FPS | 延遲 mean | p99 | 功耗 | 即時餘裕 |
|---|---|---|---|---|---|
| PyTorch FP32 + TTA（原專案） | 2.82 | 349.1 ms | 362.2 | 19.7 W | 0.56× ❌ |
| PyTorch FP32，TTA 關 | 6.21 | 155.8 ms | 163.0 | 19.3 W | 1.24× ✅ |
| **TensorRT FP16** | **11.03** | **85.6 ms** | 96.8 | 17.8 W | **2.21× ✅** |

關 TTA 2.20× × TensorRT 1.78× = **總計 3.91×**

> ⚠️ **陷阱**：在 TensorRT engine 上開 `augment=True` 的數字與關閉時完全相同
> （11.04 vs 11.03 FPS）—— TTA 被靜默忽略。engine 輸入是固定的 (1,3,960,960)，
> 而 TTA 需要多尺度推論。遷移到 TensorRT 等於放棄 TTA，是取捨不是免費加速。

---

## Phase 2：多路串流擴充性 ✅ 完成

| 串流數 | 聚合 FPS | 最慢一路 | 延遲 mean | p99 | GPU | 功耗 | FPS/W | 即時 |
|---|---|---|---|---|---|---|---|---|
| 1 路 | 11.03 | 11.03 | 85.6 ms | 96.8 | 52% | 17.8 W | 0.62 | 2.21× ✅ |
| **2 路** | **12.09** | **6.04** | 159.9 ms | 223.8 | 82% | 21.8 W | 0.56 | **1.21× ✅** |
| 3 路 | 12.19 | 4.01 | 239.7 ms | 314.4 | 85% | 22.7 W | 0.54 | 0.80× ❌ |
| 4 路 | 12.25 | 2.99 | 319.3 ms | 459.9 | 88% | 23.3 W | 0.53 | 0.60× ❌ |

**結論：單台可維持即時的容量 = 2 路。**

聚合吞吐從第二路起停在 12.1–12.3 FPS，第三、四路各只多 0.8% / 0.5%，
但延遲線性攀升 85.6 → 319 ms。**吞吐平台化 + 延遲線性惡化 = 典型飽和特徵。**

擴充效率 = 12.09 ÷ (11.03 × 2) = **54.8%**

> **最實用的一課：餘裕不等於容量。** 單路有 2.21× 即時餘裕，直覺上該塞得下兩路還有剩，
> 實際只剩 1.21×。原因在 GPU 使用率 —— 單路時已 52%，第二路推到 82%。


---

## Phase 3：啟用閒置硬體 ⏳ 進行中（2026-08-19）

### 3-A. 先量測，再決定優先序

啟用硬體前先問：這塊硬體針對的是時間預算的哪一部分？

| 單幀時間拆解 | ms | 佔比 |
|---|---|---|
| GPU 推論 | 85.6 | 94.1% |
| CPU 解碼 (OpenCV) | 5.0 | 5.9% |

→ **NVDEC 硬解的理論天花板只有 5.9%**，而 GPU 已在 82–88%。
直覺上「該用硬體解碼」是對的，但量完才知道它不是重點。優先序改為 INT8 > DLA > NVDEC。

（`ffmpeg` 的 `h264_nvv4l2dec` 已確認可用；pip 版 OpenCV 無 GStreamer 支援，
若要走 NVDEC 需改用 GStreamer Python 綁定，已確認可用。）

### 3-B. DLA 卸載 ❌ 對此模型不適用

層支援度分析（TensorRT `can_run_on_DLA`）：

| | 層數 | 佔比 |
|---|---|---|
| 可跑 DLA | 648 | 93.4% |
| 需回 GPU | 46 | 6.6% |

不支援型別：CONSTANT ×24、SHUFFLE ×9、MATRIX_MULTIPLY ×4（C2PSA 注意力）、SLICE ×4、CAST ×4

實際建置後（`trtexec --useDLACore=0 --allowGPUFallback`，engine 120 MB）：

| 裝置 | 延遲 | 吞吐 |
|---|---|---|
| GPU FP16 | 85.6 ms | 11.7 FPS |
| **DLA0 FP16** | **909 ms** | **1.1 FPS** |

**DLA 比 GPU 慢 10.6 倍。**

診斷：11 個實際回退的層**全部集中在 `model.23` 偵測頭**（Add/Concat/Sigmoid/Softmax/dfl），
是網路末端的連續區塊 → 只有一次 DLA→GPU 交接，**沒有來回震盪**。
所以 909 ms 是 DLA 真實的計算時間，不是切換開銷。

另有硬性限制：輸出 18900 與 14400 anchors 超過 DLA 單維上限 8192，偵測頭必然回退。

> **結論**：Orin NX 的 DLA 是為能效設計的，面對 57M 參數 @960² 的 YOLO11x 吃不消。
> DLA 不是「免費的額外算力」—— 它有適用的模型尺寸範圍。
> 待驗證：對 YOLO11n @640 這類小模型，DLA 是否才划算。

### 3-C. 原生 TensorRT 推論器（順帶收穫）

DLA 與 INT8 engine 都沒有 ultralytics 的 metadata 標頭，`YOLO()` 載不了 →
寫了 `trt_runner.py` 直接走 TensorRT API，NMS 用 torchvision 在 GPU 上做。

副作用是拿掉了 ultralytics 的 Python 包裝開銷：

| 推論路徑 | FPS | 延遲 |
|---|---|---|
| ultralytics + engine | 11.03 | 85.6 ms |
| **原生 TRTRunner + 同一 engine** | **11.52** | **81.6 ms** |

**+4.4%**，偵測數一致（229 vs 228）。三種 engine 之後統一走這條路徑才公平。

### 3-D. INT8 量化 ❌ 精度崩潰，不可用

校正集：210 張真實雞舍影格，跨 0700→1718 七個時段（涵蓋清晨到傍晚光照變化）。
校正器改用 torch 配置裝置記憶體（這台沒有 pycuda）。
engine 61 MB（FP16 的一半），校正 13 分鐘 + 建置 775 秒。

**速度確實變快：**

| | FPS | 延遲 | 功耗 | FPS/W | 偵測數 |
|---|---|---|---|---|---|
| FP16 | 11.52 | 81.6 ms | 18.2 W | 0.63 | 229 |
| INT8 | **18.69** | **48.3 ms** | 15.5 W | **1.21** | **82** ⚠️ |

1.62× 加速、1.92× 能效 —— **但漏掉 64% 的雞。**

以 FP16 為基準（IoU≥0.5 視為同一目標），25 幀比對：

| 指標 | 值 |
|---|---|
| INT8 recall | **36.7%** |
| INT8 precision | 91.0% |
| F1 | 0.523 |

precision 高、recall 低 → 第一直覺是「信心分數被壓縮」，降門檻應可救回。**但掃描證明不行：**

| INT8 conf | 偵測數 | recall | precision | F1 |
|---|---|---|---|---|
| 0.49 | 116 | 40.8% | 90.2% | **0.562** |
| 0.25 | 159 | 46.0% | 69.8% | 0.554 |
| 0.10 | 299 | 51.7% | 51.1% | 0.514 |
| 0.05 | 346 | 53.6% | 39.8% | 0.457 |

即使 conf=0.05、偵測數 346（比 FP16 的 250 還多），recall 也只到 53.6%，precision 崩到 39.8%。
F1 在原門檻反而最高。

> **結論：INT8 不是信心分數被壓縮，是特徵提取被量化破壞了 —— 它在偵測「不同的東西」。**
> 1.62× 的速度換掉一半的偵測，這筆交易不能做。

### 3-E. 混合精度 INT8 ❌ 補救失敗

假設：損害集中在 YOLO11x 的 **C2PSA 注意力區塊**（softmax + matmul，動態範圍寬）與偵測頭。
作法：138 層（19.9%）強制 FP16，其餘走 INT8。engine 68 MB。

| | 偵測數 | recall | precision | F1 |
|---|---|---|---|---|
| 純 INT8 | 98 | 36.7% | 91.0% | 0.523 |
| 混合精度 | 91 | **34.4%** | 90.4% | 0.498 |
| FP16 基準 | 235 | — | — | — |

**沒有改善，甚至略差。假設錯了** —— 損害不集中在注意力層。

踩坑：`CONSTANT` / `GATHER` / `CAST` 層可能帶 Int64 權重，設 FP16 會報
`cannot use precision Half with weights of type Int64`，需排除。

### 3-F. 找到真正的原因：小物件

改測「漏檢與物件大小的關係」，以 FP16 偵測框的面積分桶：

| 尺寸級距 | 數量 | INT8 recall |
|---|---|---|
| 極小 <900 px² | 882 | **10.7%** |
| 小 900–2500 | 2605 | 32.2% |
| 中 2500–6400 | 1205 | 71.5% |
| 大 >6400 | 141 | **82.3%** |

**大物件 INT8 幾乎沒問題（82%），小物件幾乎全滅（10.7%）。**

這一次解釋了全部三件事：
1. 為什麼混合精度沒用 —— 損害在 backbone 早中期承載細粒度紋理的特徵圖，不在注意力層
2. 為什麼降 conf 沒用 —— 特徵已經消失，不是被壓抑
3. 為什麼**這個應用天生對量化不友善** —— 雞舍畫面就是滿場又小、又密、又低對比的目標
   （白雞在淺色墊料上），正是 INT8 的 256 個量化級距撐不住的情境

> 可行的後續：QAT（量化感知訓練）而非 PTQ；或若相機拉近讓目標變大，INT8 就會變得可用。

### 3-G. GPU + DLA 並行 ❌ 反而更慢

最後測「至少把閒置的 DLA 撿來用」：

| 配置 | 聚合 FPS | 功耗 |
|---|---|---|
| GPU 單獨 | 10.87 | 22.0 W |
| DLA0 單獨 | 0.99 | 14.9 W |
| **GPU + DLA0 並行** | **8.95** | 24.0 W |

**淨增益 −17.7%。** GPU 從 10.87 掉到 8.02，DLA 只貢獻 0.93。

原因：DLA engine 的偵測頭本來就要回退 GPU、共用 LPDDR5 記憶體頻寬、功耗逼近上限。
**DLA 從 GPU 偷走的比它貢獻的多。**

---

## Phase 3 總結：三個嘗試，兩個被證據否決

| 手段 | 結果 | 保留？ |
|---|---|---|
| 原生 TRTRunner（去 ultralytics 開銷） | +4.4% | ✅ 保留 |
| DLA 卸載 | 慢 10.6×；並行 −17.7% | ❌ 否決 |
| INT8 量化 | 快 1.62× 但 recall 36.7% | ❌ 否決 |
| 混合精度 INT8 | recall 34.4%，無改善 | ❌ 否決 |
| NVDEC 硬解 | 只針對 5.9% 時間預算 | 🟡 降級 |

**最終保留配置：TensorRT FP16 + 原生推論器 = 11.52 FPS 單路。**

「還沒動用的硬體」查完之後的結論是：它們閒置是有原因的。
這個結論本身比硬湊一個加速比有價值 —— **量測的目的是做決定，不是產出好看的數字。**


---

## Phase 4：尚未執行的方向

| 手段 | 預期效果 |
|---|---|
| QAT（量化感知訓練） | 小目標場景下 INT8 唯一實際可行的路（見 3-F） |
| DLA 跑小模型（YOLO11n @640） | 驗證 DLA 在適用尺寸範圍內是否划算 |
| 前處理搬上 GPU + EfficientNMS | 650+ 目標的前後處理不再回 CPU |
| Zero-copy / NVMM buffer + nvv4l2decoder | Jetson CPU/GPU 共用記憶體，省 cudaMemcpy |

（已完成：關閉 TTA → Phase 1；餵食器只算一次 → cached_feeders.json；INT8 → 3-D～3-F 否決）

---

## Phase 5：場域擴充推算 ✅ 完成

雞舍配置：單棟 113m、相機間距 8m → 每棟 15 台；共 10 棟 → **全場 150 台相機**

| 項目 | 現況（集中離線） | 邊緣架構 |
|---|---|---|
| 現場裝置 | 150 台 RPi（僅錄影） | **75 台 Jetson**（即時推論） |
| 每日資料傳輸 | 1755 GB | 1.40 GB |
| 12 週飼養期 | **147 TB** | 0.12 TB |
| 資料量縮減 | — | **1250×** |
| 結果延遲 | 需搬運後批次處理 | 當下產出 |
| 現場功耗 | 450 W | 1632 W |

集中式若要不累積落後，須持續維持 **406 FPS**（全場每日 3510 萬幀，12 週 29.5 億幀），
即使用優化後速率也等於 34 台等效機器，且 147 TB 影片仍得先搬得動。

> **功耗要誠實看**：邊緣的 1632 W 比現況 450 W 高 3.6 倍。但 RPi 那 450 W 只買到錄影，
> 離線端 34 台等效機器的耗電、NAS 儲存、搬運 147 TB 的成本都不在那個數字裡。

---

## 誠實的邊界

- **偵測數比訓練集少**：本次每幀 228–321 隻，簡報訓練集標註 650–750 隻。
  推測 `conf=0.49`（沿用原設定）偏高 + 遠景小目標漏檢。NMS 負載略低於真實，但不改變飽和結論。
- **FP16 精度僅對 baseline 驗證**：統一矩陣量測中 FP16 對論文設定（含 TTA）偵測的 recall 0.87 / precision 0.97，尚未對 ground truth 重跑 P/R/F1。
- **剩餘空間**：四路時 GPU 仍只有 88%；DLA / INT8 已在 Phase 3 驗證並否決，NVDEC、GPU 前處理、engine 內 NMS 仍未實作。
- **長時間熱穩態未測**：量測窗口分鐘級，峰值 70.2°C。雞舍高溫高粉塵，連續數週的降頻行為需另驗。
- **餵食器偵測已省略**：相機固定，沿用既有 11 個座標（已驗證吻合）。相機一旦移位需重新標定。

---

## 待辦

- [x] DLA 卸載與層支援度分析（3-B、3-G）
- [x] INT8 量化 + 校正集（3-D～3-F）
- [x] 修 `EatingChickenDetect.py`：`device=3`→`0`、NAS 路徑（見 repo `research/`）
- [ ] `nvv4l2decoder` 硬體解碼取代 OpenCV
- [ ] FP16 後對 ground truth 重跑 P/R/F1

---

## 檔案清單

| 檔案 | 用途 |
|---|---|
| `chicken/export_engine.py` | TNC 模型匯出 TensorRT FP16 |
| `chicken/stream_bench.py` | **多串流 benchmark 主工具** |
| `chicken/count_pipeline.py` | 端到端計數應用（含餵食器判定、時序 CSV、標註影片） |
| `chicken/field_scaling.py` | 場域擴充推算 |
| `chicken/results_*.json` | 各設定的原始量測數據 |
| `pipeline/templates/report_template.html` | HTML 報告模板 |
| `detect.py` | 即時偵測 + FPS overlay |
| `benchmark.py` | PyTorch vs TensorRT 對比（YOLO11n 方法論驗證用） |
| `power_check.py` | 閒置 vs 負載功耗量測 |
