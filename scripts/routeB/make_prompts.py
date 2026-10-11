"""做 CosyVoice3 的阿公 prompt：挑「語調起伏最明顯」又像阿公的片段串起來（中間 0.3 秒靜音，跟 routeA 的 ref_clean 同做法）。

選段依據（routeB/eval/f0_agong_segments.json、stage1/qc_report.md）：
  agong_05  F0 std 4.62 半音、聲紋 0.873、10.94 秒
  agong_02  F0 std 3.78 半音、聲紋 0.876、6.47 秒
  agong_12  F0 std 3.26 半音、聲紋 0.858、7.82 秒
  （agong_10 起伏最大 5.34，但只有 5.4 秒、範圍 21 半音疑似有音高誤判；agong_04 起伏 4.38 但聲紋只有 0.815，不選）
另一個對照 prompt 是 routeA/sources/ref_clean_06-08-07.wav（聲紋最高的 06+08+07，語調較平穩）。
輸出（44.1 kHz）：
  routeB/prompts/expr_05-02-12.wav       去噪版（stage1/segments）
  routeB/prompts/expr_05-02-12_orig.wav  原始未去噪版（stage1/segments_orig，同切點）
"""
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
IDS = ["agong_05", "agong_02", "agong_12"]
out_dir = ROOT / "routeB" / "prompts"
out_dir.mkdir(parents=True, exist_ok=True)
for sub, suffix in [("segments", ""), ("segments_orig", "_orig")]:
    parts, sr = [], None
    for i in IDS:
        y, sr = sf.read(str(ROOT / "stage1" / sub / f"{i}.wav"), dtype="float32")
        parts += [y, np.zeros(int(0.3 * sr), np.float32)]
    y = np.concatenate(parts[:-1])
    fn = out_dir / f"expr_05-02-12{suffix}.wav"
    sf.write(str(fn), y, sr, subtype="PCM_16")
    print(fn, round(len(y) / sr, 2), "s")
