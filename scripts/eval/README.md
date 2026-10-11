# 共用評估工具（eval/）

給 seedvc（以及之後的路線 B）評估轉換結果用。全部用 **envs/prep** 的 Python 執行，在專案根目錄 `C:\Users\USER\Downloads\test` 下跑（PowerShell）：

```powershell
cd C:\Users\USER\Downloads\test
$env:PYTHONIOENCODING = 'utf-8'     # 讓中文表格在 PowerShell 正常顯示
$PY = 'C:\Users\USER\Downloads\test\envs\prep\Scripts\python.exe'
```

| 檔案 | 用途 |
| --- | --- |
| `speaker_sim.py` | 聲紋餘弦相似度（3D-Speaker CAM++，`iic/speech_campplus_sv_zh-cn_16k-common`） |
| `prosody_check.py` | 音高突然跳動的比例、特別大聲的字 |
| `baseline.json` | 阿公本人的基準數字（聲紋、prosody） |
| `audio_io.py` | 兩支工具共用的讀檔程式（不用直接執行） |
| `models/` | 聲紋模型權重快取（第一次執行自動下載，CAM++ 約 27 MB、ERes2NetV2 約 68 MB；已下載好） |

## 1. 聲紋相似度 `speaker_sim.py`

計畫書「評估方式」的做法：測試音分別跟**阿公原始錄音前半段**、**後半段**各比一次。

```powershell
# 一個或多個測試檔／資料夾 vs 阿公原始錄音前半、後半（主要指標）
& $PY eval\speaker_sim.py --agong-halves --test routeA\outputs\zeroshot.wav routeA\outputs\ft_step200\ --json routeA\eval\sim_v1.json

# 同時也跟「只去噪版」前半、後半比（切點相同；轉換結果是乾淨音時建議一起看）
& $PY eval\speaker_sim.py --agong-halves --agong-denoise-halves --test routeA\outputs\ --json routeA\eval\sim_all.json

# 自訂參考（檔案或資料夾都可以，可多個）
& $PY eval\speaker_sim.py --ref inputs\voice_reference_15s.wav stage1\segments --test routeA\outputs\a.wav

# 只給參考、不給測試 → 參考兩兩互比
& $PY eval\speaker_sim.py --agong-halves
```

選項：
- `--ref`／`--test`：檔案或資料夾（wav/flac/mp3/ogg/m4a…；資料夾預設不遞迴，加 `--recursive` 遞迴）。
- `--agong-halves`：參考加入 `stage1/wav/male_voice_only.wav` 的前半（0–48.66 秒）與後半（48.66–103.82 秒），切點是中點附近最安靜處。
- `--agong-denoise-halves`：參考加入 `stage1/enhanced/male_voice_only_denoise.wav` 的前後半（同切點）。
- `--halves 檔案`：任意檔案切前後半當參考。
- `--model campplus|eres2netv2`：預設 `campplus`；`eres2netv2`（`iic/speech_eres2netv2_sv_zh-cn_16k-common`）可當第二意見。
- `--device auto|cuda|cpu`：預設 auto。CAM++ 很小，CPU 也只要幾秒；**seedvc 正在訓練時請用 `--device cpu`**，避免搶顯存。
- `--json 路徑`：另存完整結果（模型資訊、每個參考的來源與時間範圍、分數矩陣、對所有參考的平均）。

輸出表格：列＝測試音，欄＝各參考，最後一欄「平均」＝對所有參考的平均。報告時請寫「對原始前半／後半各多少、平均多少」。

### 怎麼判讀（數字詳見 `baseline.json` 的 `speaker`、`speaker_note`）

| 比較 | CAM++ |
| --- | --- |
| 阿公原始前半 vs 後半（同一人基準） | **0.939**（之前雲端量的 0.92，很接近） |
| voice_reference_15s vs 原始錄音其餘部分 | 0.906 |
| 只去噪版 vs 原始（不同半） | 0.797；把原始雜音加回去 → 0.892 |
| 主力片段（去噪，5–11 秒）vs 原始錄音另一半，平均 | 0.709（0.61–0.79） |
| 完整增強版 vs 原始（不同半） | 0.528（增強改了音色，沒採用） |
| 之前 kNN-VC 轉換結果 | 0.60–0.67 |

- 我們的同一人基準（0.939）跟之前的 0.92 差不到 3%，**0.67 門檻可直接沿用**；要嚴格按比例換算的話是 0.67 × 0.939 / 0.92 ≈ **0.684**。
- 要注意：聲紋模型會把「雜音／收音環境」也算進去。乾淨的阿公聲音跟有雜音的原始錄音比，分數就只有 0.71–0.80。所以乾淨的轉換結果，對原始錄音前後半的分數**實際上限大約 0.8**；看到 0.75–0.8 已經接近「真阿公、乾淨版」的水準。同時看 `--agong-denoise-halves` 的分數（去噪版自己前後半互比是 0.901）比較公平。
- 測試音太短（< 3 秒）分數會偏低、不穩；盡量用同一段 20–40 秒的台語稿比較各版本。

## 2. 順不順 `prosody_check.py`

```powershell
# 一個或多個檔案／資料夾；--baseline 會一併列出阿公本人的參考值並判斷是否「不高於阿公」
& $PY eval\prosody_check.py routeA\outputs\ft_step200\ --baseline eval\baseline.json --json routeA\eval\prosody_step200.json
```

指標（預設參數，要跟 baseline 一樣才能比）：
- **音高跳動比例** `pitch_jump_ratio`：相鄰兩個有聲幀（每 10 ms）之間，音高變化超過 3 個半音的比例。有聲判斷用 librosa.pyin，幀間變化用逐幀獨立的 librosa.yin（pyin 的平滑會把破音抹掉）。自然語音幀間變化多在 1 半音內，超過 3 半音的大多是破音／八度跳。
- **特別大聲的字** `loud_words`／`loud_per_10s`：100 ms 短時響度比「有聲部分響度中位數」高 8 dB 以上的峰（峰間 ≥ 150 ms、突出 ≥ 3 dB）；`loud_per_10s` 是每 10 秒有聲時間的個數，不同長度的檔案才能比。
- 多檔「合計」：跳動比例＝總跳動數／總有聲幀對數；大聲字＝總峰數／總有聲秒數 × 10。

阿公本人參考值（`baseline.json` → `prosody.agong_reference`，主力版 12 段合計 93.4 秒）：

| 指標 | 阿公（合計） | 單段中位數 | 單段 P90 |
| --- | --- | --- | --- |
| 音高跳動比例 | 0.0156 | 0.0188 | 0.0316 |
| 每 10 秒大聲字 | 2.06 | 1.62 | 4.37 |

判讀：計畫書寫「不高於阿公本人錄音的水準」→ 合計值 ≤ 上表「合計」。單一短檔波動大，請用合計或同一段稿比較。

## 3. 其他

- 第一次執行會從 modelscope 下載聲紋模型到 `eval/models/`；modelscope 連不上時 CAM++ 會自動改從 Hugging Face `funasr/campplus` 下載同一份權重。
- 讀 m4a 等 soundfile 不支援的格式時會呼叫 `tools/FFMPEG_PATH.txt` 裡的 ffmpeg。
- 重算 baseline：`& $PY stage1\scripts\make_baseline.py`（只用 CPU）。
