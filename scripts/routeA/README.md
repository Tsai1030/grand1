# 路線 A：Seed-VC 聲音轉換（台語原音 → 阿公的聲音）

完整結果與建議見 **`eval_report.md`**；試聽請直接開 **`listen/`**（有 README）。

## 一個指令重新轉換（換了新的台語原音時用）

1. 把新的 `taiwanese_source_joined.wav` 放進 `inputs\`（或專案根目錄）。也可以用 `-Source` 指定任何 wav。
2. 在 PowerShell 執行：

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\USER\Downloads\test\routeA\convert_final.ps1
# 指定別的原音（例如 routeB 產生的）：
powershell -ExecutionPolicy Bypass -File C:\Users\USER\Downloads\test\routeA\convert_final.ps1 -Source C:\Users\USER\Downloads\test\routeB\xxx.wav
```

它會：
- 用 `best_checkpoints.json` 列的版本（評估後最好的 1–3 個存檔點＋各自的參考音、前處理）逐一轉換；
- 自動跑評估：聲紋相似度（對阿公原始錄音、去噪版的前後半）、音高跳動／大聲字、F0 相關（聲調風險）、語速與停頓；
- 結果放在 `routeA\final\<日期_時間>\`：每個版本一個 wav，數字在 `summary.tsv`。

其他選項：`-Ref <wav>`（全部改用指定參考音）、`-OutDir <資料夾>`、`-Jobs <json>`（自訂版本清單）、`-NoEval`（只轉不評估）。
每個版本約 1 分鐘（RTX 3070，含載入模型）；評估在 CPU 上約 2–3 分鐘。

判讀：`sim_dn_mean`（對阿公去噪版前後半平均）越高越像，同一人基準 0.901；`f0_r_global`（F0 輪廓跟原音的相關）
≥ 0.70 表示聲調大致照原音走（聲調風險低），< 0.45 聲調風險高。數字只用來篩掉明顯不好的版本，最後以家人試聽為準。

## 單檔轉換（給其他腳本呼叫）

`scripts\run_infer.py` 是 Seed-VC `inference.py`／`inference_v2.py` 的包裝：固定亂數種子、輸出到指定檔名，並另存 `<輸出>.json`（時間、顯存）。

```powershell
$PY = 'C:\Users\USER\Downloads\test\envs\seedvc\Scripts\python.exe'
$RA = 'C:\Users\USER\Downloads\test\routeA'
# 最佳設定：v1 去噪版微調 100 步＋去噪參考音（建議先把來源的長停頓縮到 0.45 秒，見下）
& $PY $RA\scripts\run_infer.py --model v1 --out D:\out\agong.wav --seed 1234 -- `
    --source D:\in\source.wav --target $RA\sources\ref_clean_06-08-07.wav `
    --checkpoint $RA\runs\v1_denoise\DiT_epoch_00016_step_00100.pth `
    --config $RA\runs\v1_denoise\config_dit_mel_seed_uvit_whisper_small_wavenet.yml `
    --diffusion-steps 100 --inference-cfg-rate 0.7 --f0-condition False --auto-f0-adjust False --fp16 True

# 縮短停頓（字不變速、不改音高）
& $PY $RA\scripts\make_variants.py --source D:\in\source.wav --pause-cap 0.45 --out D:\in\source_pausecap0.45.wav

# v2（零樣本或微調過的 CFM），convert-style 開／關
& $PY $RA\scripts\run_infer.py --model v2 --out D:\out\v2.wav -- --source D:\in\source.wav --target <參考音> `
    --cfm-checkpoint-path $RA\runs\v2_denoise\CFM_epoch_00083_step_00500.pth --convert-style false `
    --diffusion-steps 30 --intelligibility-cfg-rate 0.7 --similarity-cfg-rate 0.7
```

`--` 之後的參數原封不動交給 Seed-VC（見 `repos\seed-vc\README.md`）；`--output` 不用給。來源與參考音請給絕對路徑。

## 資料夾

| 路徑 | 內容 |
| --- | --- |
| `eval_report.md` | 評估報告（版本 × 指標、建議、實測時間顯存、Windows 問題） |
| `listen/` | 試聽包（A_… F_，附 README） |
| `convert_final.ps1`、`best_checkpoints.json` | 一個指令重新轉換＋評估 |
| `zero_shot/` | 零樣本基準（v1 擴散 30／50 步、v2 style 開／關） |
| `finetune/<run>/` | 每個微調存檔點轉同一段原音的結果＋`summary.tsv`＋`eval/` |
| `experiments/` | 參考音、停頓縮短／變速、亂數種子變異等實驗（`jobs_round*.json` 是設定） |
| `experiments_metal/` | 金屬感：只改擴散步數、fp16、cfg、44.1 kHz f0 模型的對照（`jobs_metal.json`） |
| `runs/<run>/` | 微調存檔點（每 100 步一個）、`train.log`、`train_stats.json`（時間、顯存） |
| `sources/` | 參考音與來源變體（`ref_clean_06-08-07.wav`、`src_pausecap0.45.wav`…；`TEMP_*` 是早期暫用來源，不要用） |
| `eval/` | 零樣本與其他彙整用的評估 JSON、`tables.md` |
| `scripts/` | 所有腳本（見下） |
| `setup/` | 建環境：`setup_env.ps1`、`requirements-seedvc-win.txt` |
| `logs/` | 執行紀錄 |

腳本：`run_infer.py`（單檔轉換）、`train_keepall.py`／`run_train.ps1`（微調、保留每個存檔點）、`convert_and_eval.py`（批次轉換＋評估）、
`f0_corr.py`（聲調風險）、`rhythm.py`（語速／停頓）、`hf_quality.py`（高頻乾淨度）、`make_variants.py`（去噪參考音、停頓縮短、變速）、`make_tables.py`（報告表格）、
`prefetch_weights.py`（預先下載權重）、`run_zero_shot.ps1`。

## 環境

`envs\seedvc`：Python 3.10.10、torch 2.6.0+cu124、transformers 4.46.3，Seed-VC commit `51383ef`（`repos\seed-vc`）。
重建：`powershell -ExecutionPolicy Bypass -File routeA\setup\setup_env.ps1`，再 `envs\seedvc\Scripts\python.exe routeA\scripts\prefetch_weights.py` 下載權重（約 3.8 GB，放在 `repos\seed-vc\checkpoints\`）。
評估工具用 `envs\prep`（見 `eval\README.md`）。
