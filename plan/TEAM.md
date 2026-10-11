# 團隊協作說明（阿公台語語音：階段一＋路線 A）

計畫書：`阿公台語語音影片：開源模型自建計畫書.md`（根目錄，執行期間不要改它；由 lead 最後統一回填）。
本次範圍：階段一（阿公音檔整理）＋路線 A（Seed-VC 聲音轉換：零樣本＋微調＋評估）。不做 CosyVoice3 微調、不做影片。

## 機器
- Windows 11，RTX 3070 8GB（桌面已佔約 1.2GB），磁碟剩約 300GB。
- Python：用 `py -3.10`（3.10.10）。PATH 上的 `python` 是 Microsoft Store 假捷徑，不要用。
- 沒有 conda、uv、ffmpeg（ffmpeg 由 audio-prep 安裝，見下）。git 有。

## 目錄
```
inputs/                 原始素材（唯讀，不可修改或刪除）
  male_voice_only.mp3     阿公 1 分 44 秒
  voice_reference_15s.wav 阿公 14 秒參考音
  （使用者之後會放：taiwanese_source_joined.wav、照片、祝福文；也可能放在根目錄）
tools/ffmpeg/           ffmpeg（audio-prep 安裝；完整路徑寫在 tools/FFMPEG_PATH.txt）
envs/prep/              階段一＋評估工具用的 venv
envs/seedvc/            Seed-VC 用的 venv
repos/seed-vc/          Seed-VC 原始碼
stage1/                 audio-prep 的產出
  segments/               最終阿公片段（1–30 秒），也是 Seed-VC 微調資料
  manifest.tsv            一段一行：檔名、語者、秒數、聲紋相似度、華語意思(ASR)、台語漢字(待填)、台羅(待填)
  qc_report.md
  STAGE1_DONE.txt         寫入代表階段一完成、GPU 交給 seedvc
eval/                   共用評估工具（audio-prep 建立，seedvc 使用）
  speaker_sim.py          聲紋相似度
  prosody_check.py        音高跳動比例、過大聲的字
  baseline.json           阿公前後半段互比等基準
  README.md               用法
routeA/                 seedvc 的產出
docs/                   doc-research 的產出；計畫書_原始備份.md 是原稿備份
```

## 規則
1. **GPU 一次只給一個重工作**：在 `stage1/STAGE1_DONE.txt` 出現前，GPU 屬於 audio-prep；之後屬於 seedvc。doc-research 不用 GPU。
2. 長時間安裝、下載、訓練用背景執行或長 timeout，不要卡住。
3. 不用付費服務、不註冊帳號、不需要 HF token。需要 gated 模型的步驟改用不需授權的替代品，並在報告註明。
4. 給使用者看的報告、清單備註一律用繁體中文。
5. 數字要可重現：報告寫明用了哪個模型、哪個檔案、哪些參數。
