# 路線 B 報告：開源 TTS 台語 ＋（可選）Seed-VC

## 方法
- 文字：8 句（第 1 句已確認是「我是阿公。」，ASR 全部聽成「我是阿公」）。
- CosyVoice3-0.5B-2512 instruct2，4 組（base×expr、warm×expr、warm×clean22、warm×expr_orig），每句 3 seed，GPU 生成。
- OmniVoice voice design（"male, elderly, moderate pitch"）＋台羅文字；rate 3.2（原速）與 2.4（慢速，本次新增，GPU 推論 31 秒）。
- 評估：`routeB/scripts/eval_candidates.py`（Breeze-ASR-26 用 CPU、beam 3；tone_dtw 對台灣腔原音；f0_stats；CAM++ 聲紋）；每方法每句挑分數最高 seed，concat.py 串接（停頓 0.6–1.0 秒），passage_eval.py 評整段。數據：`routeB/eval/candidates.json`、`candidates_ov_slow.json`、`passage_final.json`。

## 結果（整段）
| 方法 | ASR 聽懂 | tone_r | F0 範圍 st（阿公 10.1） | 字/秒 | 聲紋 |
|---|---|---|---|---|---|
| CV base×expr | 7/8 | 0.53 | 10.0 | 2.67 | 0.66 |
| CV warm×expr | 7/8 | 0.47 | 9.1 | 2.73 | 0.65 |
| CV warm×clean22 | 6/8 | 0.51 | 7.7 | 2.67 | 0.66 |
| CV warm×expr_orig | 7/8 | 0.48 | 9.6 | 2.65 | 0.58 |
| OV design 原速 | 8/8 | 0.59 | 10.3 | 3.11（太快） | 0.15 |
| OV design 慢速 | 8/8 | 0.43 | 10.8 | 2.66 | 0.17 |
| OV 慢速 → Seed-VC | （沿用） | – | 8.5 | 2.69 | 0.88 |
| CV base → Seed-VC | （沿用） | – | 8.7 | 2.66 | 0.90 |

## 結論
- 口音：OmniVoice（台羅輸入）最像台灣台語，ASR 8/8 且 24 個 seed 中 23 個聽懂；CosyVoice3 只有 14–16/24 個 seed 聽懂，「闽南话」指令偏中國腔、常把台語詞唸成華語。
- 聲調：OV 風險低；慢速降到 tone_r 0.43（慢速拉長改變了走向）。CV 較高風險。
- 韻律：兩者 F0 範圍都接近阿公（約 10 st）；OV 中位數 180 Hz 偏高，經 Seed-VC 降到 130 Hz，但 F0 範圍縮到 8.5 st。
- 聲音：OV 本身不像阿公（0.17）；經 Seed-VC 後 0.88，接近同一人基準 0.901。CV 本身 0.66，經 Seed-VC 0.90。
- 推薦：試聽 C（OV 慢速 → Seed-VC）。若覺得口音可以再放寬、想要更像阿公的聲紋，聽 D。Seed-VC 後未重跑逐句 ASR，建議人工聽 C 確認字沒被轉壞。

## Windows 問題
- Breeze-ASR 在 GPU 被訓練占用時只能 CPU（約 1 秒/句，可用）。
- OmniVoice 用阿公參考音但逐字稿留空會唸壞，改用 voice design；Seed-VC 需在 repo 目錄執行（run_infer.py 已處理）。
- 路徑含中文／編碼：腳本需 PYTHONIOENCODING=utf-8。python 要用 py -3.10 或 envs 內的 python。

## 建議
1. 以 C 為主，同時聽 B、D 比較。2. 想更像阿公的語調：用阿公台語逐字稿微調（見下）。3. 句間停頓、語速已符合規定。

## 評估：用阿公片段＋台語逐字稿做 zero-shot prompt 或 LoRA（不實作）
- **CosyVoice3 zero-shot prompt**：需要 (1) 一段 3–15 秒乾淨阿公片段（stage1/segments 已有，ref_clean_06-08-07 約 22 秒需剪短）；(2) 該段的正確逐字稿——要是台語漢字或華語意思皆可，但模型的文字前端是華語／閩南語混合，台語漢字／台羅逐字稿未必被正確處理；(3) 在 zero-shot 模式下口音由 prompt 決定，阿公的口音是台灣腔，理論上會比指令式更像台灣腔，也能學到音色與一些語調。需要人工聽寫那幾段（台羅）。風險：prompt 逐字稿與語言不符時 CosyVoice 會唸壞（跟 OmniVoice 同類問題）；吞吐小、可行、成本低。建議先試 3 段各 5–10 秒。
- **OmniVoice LoRA 微調**：需要 (1) 阿公 93 秒（切成 1–30 秒共約 12 段）全部有人工校正的台羅／漢字逐字稿；(2) 訓練腳本（官方是否提供 LoRA 流程要先確認，否則要自己接 PEFT）；(3) 顯存：模型較小，8GB 的 RTX 3070 可能夠，但 93 秒資料極少，容易過擬合、聲音變怪或口音退化。可行性中低：資料量不足為主要風險，建議先做 zero-shot prompt（有台羅逐字稿的參考音）看效果，再考慮 LoRA，並保留部分片段做驗證。
