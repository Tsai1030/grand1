# 階段一：阿公音檔整理 品質報告

2026-10-11 · audio-prep

**結論**：主力版本選「只去噪」（resemble-enhance denoise）。切成 13 段，剔除 1 段（疑似混到別人或電視聲），保留 **12 段、共 93.38 秒**，放在 `stage1/segments/`；同切點的原始未增強版放在 `stage1/segments_orig/`，兩版都可以拿去微調比較。

## 1. 工具與版本

| 項目 | 版本／來源 | 用途 |
| --- | --- | --- |
| Python | 3.10.10，venv `envs/prep` | |
| PyTorch／torchaudio | 2.6.0+cu124（RTX 3070，CUDA 可用） | |
| ffmpeg | 7.1 essentials（gyan.dev build，取自 PyPI `imageio-ffmpeg` 0.6.0 wheel，SHA256 已核對）→ `tools/ffmpeg/bin/ffmpeg.exe` | mp3 → wav |
| resemble-enhance | GitHub repo commit `8e97814`（`repos/resemble-enhance`），權重 HF `ResembleAI/resemble-enhance` enhancer_stage2 | 去噪、增強 |
| 3D-Speaker | GitHub repo commit `065629c`（`repos/3D-Speaker`，只用模型定義） | 聲紋 |
| 聲紋模型 | CAM++ `iic/speech_campplus_sv_zh-cn_16k-common` v1.0.0（主要）；ERes2NetV2 `iic/speech_eres2netv2_sv_zh-cn_16k-common` v1.0.1（第二意見），modelscope 下載、不需授權 | |
| silero-vad | 6.2.3 | 找停頓 |
| pyloudnorm | 0.2.0（ITU-R BS.1770-4） | 響度統一 |
| librosa | 0.11.0（pyin／yin） | prosody 檢查 |
| Breeze-ASR-26 | HF `MediaTek-Research/Breeze-ASR-26` commit `7b99268`，transformers 4.57.6，fp16、GPU | 華語意思 |
| 其他 | numpy 2.2.6、scipy 1.15.3、soundfile 0.14.0、soxr 1.1.0、modelscope 1.40.2 | |

所有腳本在 `stage1/scripts/`，可重跑：`run_enhance.py` → `compare_versions.py` → `noise_back_test.py` → `segment_and_qc.py --main denoise` → `run_asr.py` → `make_baseline.py` → `make_manifest.py`。

## 2. 原始音檔

- `inputs/male_voice_only.mp3`：44.1 kHz 單聲道、192 kbps、103.82 秒。轉成 `stage1/wav/male_voice_only.wav`（32-bit float，保留 44.1 kHz）。
- 整段響度 -19.4 LUFS、峰值 -2.2 dBFS。約 11% 的取樣是數位靜音（之前去掉旁人聲音時清成 0 的地方）。
- `inputs/voice_reference_15s.wav` 是原始錄音 **37.05–50.47 秒**那段，中間兩處停頓略剪短（分段互相關 0.99 定位）。

## 3. 去噪／增強與選版

resemble-enhance 三種處理，都輸出 44.1 kHz、跟原始錄音逐點對齊（`stage1/enhanced/`）：
- `male_voice_only_denoise.wav`：只去噪（等同 CLI `--denoise_only`）
- `male_voice_only_enhance.wav`：完整增強（CLI 預設：nfe=64、midpoint、lambd=1.0、tau=0.5、seed 1234）
- `male_voice_only_enhance_lambd0.5.wav`：較輕的增強（lambd=0.5，只供比較）

| 版本 | 聲紋：同內容 vs 原始 | 聲紋：不同半 vs 原始 | 語音 RMS | 非語音 RMS | 粗估 SNR |
| --- | --- | --- | --- | --- | --- |
| 原始 | 1.000 | **0.939**（前半 vs 後半） | -17.9 dBFS | -30.8 dBFS | 12.9 dB |
| **只去噪** | 0.849 | 0.797 | -18.5 | -61.8 | **43.4 dB** |
| 完整增強 | 0.557 | 0.528 | -17.5 | -42.5 | 25.0 dB |
| 增強 lambd 0.5 | 0.608 | 0.569 | -18.2 | -47.9 | 29.7 dB |

（聲紋為 CAM++；ERes2NetV2 結果幾乎一樣，見 `stage1/version_compare.json`。SNR＝VAD 語音區 vs 非語音區的 RMS 差，非語音區前後各留 0.1 秒不算。）

**為什麼選「只去噪」**：
1. **完整增強明顯改了音色**：跟原始只剩 0.53–0.56，比之前 kNN-VC 的轉換結果（0.60–0.67）還低。增強器會把聲音「重畫」一遍，正是計畫書提醒的「增強太重會改變音色」。lambd 降到 0.5 也只有 0.57–0.61。
2. **只去噪的分數下降，主要是雜音被拿掉，不是音色被改**：把原始錄音的背景雜音（非語音區段）加回只去噪版，不同半的分數從 0.797 回到 **0.892**，同內容從 0.84 回到 0.94（`stage1/noise_back_test.json`）。聲紋模型本來就會把收音環境算進去。
3. **雜音大幅降低**：非語音區從 -30.8 降到 -61.8 dBFS（約 31 dB）。原始錄音戶外、離手機遠，直接拿去微調，Seed-VC 會連雜音一起學。

原始版仍保留成 `stage1/segments_orig/`，請 seedvc 兩版都微調比較（計畫書風險表的要求）。

## 4. 切句

- 在只去噪版上跑 silero-vad（threshold 0.5、最短靜音 150 ms、最短語音 150 ms），找到 54 段語音；停頓大多 0.2–0.9 秒，較長的停頓在 65.2 秒（1.7 秒）、95.2 秒（1.2 秒）、99.9 秒（1.3 秒）。
- 切點規則：停頓 ≥ 0.8 秒優先切；每段語音範圍至少 3 秒；超過 12 秒就在段內最長、最靠中間的停頓再切；語音前留 0.25 秒、後留 0.30 秒（不超過停頓中點）。
- **同樣切點**同時切主力版（`segments/`）和原始版（`segments_orig/`）。
- 響度：每段用 pyloudnorm 統一到 **-20 LUFS**，用 4 倍超取樣估真峰值，限制在 -1 dBFS 以下、不削峰（agong_02 因此停在 -20.8 LUFS，仍在計畫書 -23 到 -18 範圍內）。存成 44.1 kHz、16-bit PCM。

## 5. 片段清單（保留 12 段，共 93.38 秒）

聲紋相似度＝該段跟「其他所有片段的平均聲紋」（不含自己）的餘弦相似度，CAM++。

| 檔名 | 原始錄音時間 | 秒數 | 聲紋（主力／原始版） | 響度 LUFS | 真峰值 dBFS | 華語意思（Breeze-ASR-26，僅供參考） |
| --- | --- | --- | --- | --- | --- | --- |
| agong_01 | 0.00–11.38 | 11.38 | 0.887／0.887 | -20.0 | -3.9 | 轎子 用柱子 柱子 綁在上面釘木板 走路就這樣 沒轎子 只有木板 釘在上面這樣 |
| agong_02 | 11.40–17.87 | 6.47 | 0.876／0.870 | -20.8 | -1.0 | 走路這樣 才會移動 哪裡都能騎 |
| agong_03 | 17.87–24.08 | 6.20 | 0.773／0.795 | -20.0 | -4.1 | 不然怎麼會這麼高 就讀書的時候 變成這個樣子 你說 |
| agong_04 | 24.42–32.11 | 7.69 | 0.815／0.794 | -20.0 | -5.0 | 又難抽 剛好 放下去 草頭 草頭 選對了 |
| agong_05 | 32.26–43.20 | 10.94 | 0.873／0.884 | -20.0 | -3.9 | 要造座位 整個都去學校 大家穿鞋子 駱仔 對 駱仔 駱仔在載小孩 |
| agong_06 | 43.21–50.64 | 7.43 | 0.917／0.906 | -20.0 | -3.0 | 是來大慈大悲的王鎮 當時是喪葬的人 要去警衛那裡 |
| agong_07 | 50.64–56.62 | 5.98 | 0.884／0.885 | -20.0 | -4.9 | 櫃檯上也沒幾個 一個個賣菜 |
| agong_08 | 56.74–65.45 | 8.71 | 0.894／0.866 | -20.0 | -2.8 | 妹婿又大肚子 連姓孫的也是 老婆也是同一代的 |
| agong_09 | 66.66–74.28 | 7.62 | 0.794／0.834 | -20.0 | -3.4 | 我們在哪裡 阿彌陀佛 阿彌陀佛 賣菜喔 她在外面拉拉扯扯的 |
| agong_10 | 74.50–79.90 | 5.40 | 0.869／0.870 | -20.0 | -3.6 | 他就 孩子聽到話就踩來踩去 |
| agong_11 | 79.94–87.68 | 7.74 | 0.765／0.790 | -20.0 | -3.8 | 蓋那邊就蓋 反正都… 都來這裡就蓋 我蓋那一邊就… 蓋得整整齊齊 |
| agong_12 | 87.68–95.50 | 7.82 | 0.858／0.838 | -20.0 | -2.3 | 你這麼冷 真冷 都四十八歲了 穿這種衣服也不來看我 |

逐字稿清單：`stage1/manifest.tsv`（台語漢字、台羅兩欄留空，請家人聽寫）。ASR 細節在 `stage1/asr.json`；agong_11 第一次解碼出現 Whisper 常見的無限重複幻覺，用重複懲罰（no_repeat_ngram_size=4、repetition_penalty=1.2）重跑。Breeze-ASR-26 是台語轉華語文字，錯字多，只能幫忙知道大意，台語用字一定要人工補。

## 6. 被剔除的片段

| 檔名 | 時間 | 秒數 | 聲紋（主力／原始版） | 原因 |
| --- | --- | --- | --- | --- |
| agong_13 | 96.07–103.82 | 7.75 | 0.732／0.799 | 低於門檻 0.759（12+1 段的中位數 0.869 − 3×MAD），疑似混到別人或電視聲 |

佐證：這段後半的聲紋特別低（98.9–100.1 秒只有 0.51、101.0–103.8 秒 0.61，前段 96.1–98.7 秒 0.74）；音高 P90 到 343 Hz，其他段都在 150–210 Hz；ASR 出來是「我跟曉冬的緣分 應該是從明月變成的」，不像阿公在聊的內容，比較像電視劇台詞。檔案已移到 `stage1/rejected/segments/`、`stage1/rejected/segments_orig/`，沒有刪除；**建議家人聽一下**，如果確定是阿公，可以只把 96.1–98.7 秒那一小段救回來。

另外 agong_03（0.773）、agong_11（0.765）、agong_09（0.794）分數偏低但高於門檻，保留；家人聽寫時順便確認是否都是阿公。

## 7. 基準數字（`eval/baseline.json`）

| 比較（CAM++） | 分數 |
| --- | --- |
| 阿公原始錄音前半 vs 後半（切在 48.66 秒） | **0.939**（ERes2NetV2：0.937） |
| voice_reference_15s vs 原始錄音其餘部分 | 0.906 |
| 只去噪版 vs 原始（不同半） | 0.797（加回雜音 → 0.892） |
| 完整增強版 vs 原始（不同半） | 0.528 |
| 主力片段（去噪）vs 原始錄音另一半，平均 | 0.709（0.61–0.79） |
| 原始版片段 vs 原始錄音另一半，平均 | 0.831 |
| 只去噪版自己前半 vs 後半 | 0.901 |

- 我們量到的同一人基準 0.939 跟之前雲端的 0.92 很接近（比值 1.02），**0.67 門檻可以直接沿用**；按比例換算是 0.684。
- 但要知道：乾淨的聲音跟有雜音的原始錄音比，分數會掉到 0.7–0.8。所以 Seed-VC 輸出若是乾淨的，對原始錄音前後半的分數**實際上限約 0.8**。評估時建議同時看「對去噪版前後半」的分數（`speaker_sim.py --agong-halves --agong-denoise-halves`）。
- 阿公本人 prosody 參考（主力 12 段合計）：音高跳動比例 **0.0156**、每 10 秒有聲時間大聲字 **2.06**；voice_reference_15s 本身是 0.0124、1.49。

## 8. 遇到的 Windows 問題與解法

| 問題 | 解法 |
| --- | --- |
| GitHub release 下載被限速（約 40 KB/s，BtbN ffmpeg 190 MB 要 1 小時以上）；gyan.dev 連不上 | 改從 PyPI 下載 `imageio-ffmpeg` wheel（內含 gyan.dev 的 ffmpeg 7.1 essentials 靜態執行檔），核對 SHA256 後取出 `ffmpeg.exe`。沒有 ffprobe，用不到。 |
| resemble-enhance 在 import 時就要 deepspeed（訓練用），Windows 裝不起來 | 不裝 pip 版，直接用 clone 的 repo 的 Python API；在 import 前塞一個只供 import 用的 deepspeed 假模組（推論不會呼叫到）。 |
| torch 2.6 的 `torch.load` 預設 `weights_only=True`，官方 deepspeed 格式 checkpoint 載不進來 | 只在 `run_enhance.py` 內改成 `weights_only=False`（來源是官方 HF repo）。 |
| 官方 `hparams.yaml` 存了 `pathlib.PosixPath`，Windows 無法建立 | 在 `run_enhance.py` 內把 `pathlib.PosixPath` 指向 `WindowsPath`。 |
| 完整增強用原版 30 秒一塊會超出 8GB 顯存，掉到共享記憶體，跑 12 分鐘還沒完 | 改成 10 秒一塊（重疊 1 秒），每版約 50 秒跑完，顯存峰值 3.8 GB。三版都用同一設定重跑。 |
| PowerShell 顯示中文亂碼 | 執行前設 `$env:PYTHONIOENCODING='utf-8'`。 |
| PATH 上的 `python` 是 Microsoft Store 假捷徑 | 一律用 `py -3.10` 建 venv、用 venv 內 python.exe 的完整路徑。 |

## 9. 給隊友與家人的待辦

- **家人**：聽寫 `stage1/manifest.tsv` 的台語漢字、台羅兩欄；聽一下 `stage1/rejected/segments/agong_13.wav` 是不是阿公。
- **seedvc**：微調資料用 `stage1/segments/`（主力），另外用 `stage1/segments_orig/` 微調一次比較；評估工具見 `eval/README.md`。
