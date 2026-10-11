"""把逐句檔串成整段：去掉每句頭尾靜音，句間插入停頓，整段響度統一到 -20 LUFS。不加速、不改字的長短。

句間停頓（秒）：使用者 2026-10-11 回饋「路線 A 太快（約每秒 3.1 字）」，要求句間 0.6–1.0 秒、
整段語速每秒 2.5–2.8 字（祝福語氣，要慢、要穩）。
做法：基本型 GAPS（不等長，避免每句一樣長的「念稿」感；最後一句「阿公足疼恁的。」前停最長），
再整體縮放，讓「93 字 ÷ 整段說話時間（第一個字到最後一個字）」最接近 2.65 字/秒，
但每個停頓都限制在 0.6–1.0 秒（句子本身太快時，縮放到 1.0 秒上限也可能還是 > 2.8，會在輸出時印出來）。

用法：python concat.py --pick pick.json --out 輸出.wav
  pick.json：{"files": ["...s1_seed2.wav", "...s2_seed1.wav", ...]}（照句序）
或：python concat.py --files f1.wav f2.wav ... --out 輸出.wav
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

GAPS = [0.75, 0.85, 0.70, 0.90, 0.70, 0.80, 1.00]
GAP_MIN, GAP_MAX = 0.6, 1.0
TARGET_RATE = 2.65
N_SYL = 93
SR = 24000
TARGET_LUFS = -20.0
LAST_INFO = {}


def trim(y, sr, margin=0.04):
    hop = int(sr * 0.01)
    rms = librosa.feature.rms(y=y, frame_length=hop * 3, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    thr = max(np.percentile(db, 95) - 35, np.percentile(db, 5) + 6)
    idx = np.where(db > thr)[0]
    if len(idx) == 0:
        return y
    a = max(0, idx[0] * hop - int(margin * sr))
    b = min(len(y), (idx[-1] + 1) * hop + int(margin * sr) + hop * 2)
    return y[a:b]


def fade(y, sr, ms=10):
    n = int(sr * ms / 1000)
    if len(y) > 2 * n:
        y = y.copy()
        y[:n] *= np.linspace(0, 1, n)
        y[-n:] *= np.linspace(1, 0, n)
    return y


def build(files, gaps=GAPS, n_syl=N_SYL, target_rate=TARGET_RATE):
    segs = []
    for f in files:
        y, _ = librosa.load(str(f), sr=SR)
        segs.append(fade(trim(y, SR), SR))
    speech = sum(len(s) for s in segs) / SR
    g = np.array(gaps[: len(segs) - 1], dtype=float)
    # 讓 n_syl / (speech + sum(gaps)) ≈ target_rate；每個停頓夾在 GAP_MIN–GAP_MAX
    need = n_syl / target_rate - speech
    lo, hi = 0.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        tot = np.clip(g * mid, GAP_MIN, GAP_MAX).sum()
        lo, hi = (mid, hi) if tot < need else (lo, mid)
    g = np.clip(g * (lo + hi) / 2, GAP_MIN, GAP_MAX)
    span = speech + g.sum()
    LAST_INFO.clear()
    LAST_INFO.update({"speech_sec": round(speech, 2), "gaps": [round(x, 2) for x in g], "span_sec": round(span, 2),
                      "text_rate": round(n_syl / span, 2)})
    parts = []
    for i, s in enumerate(segs):
        parts.append(s)
        if i < len(segs) - 1:
            parts.append(np.zeros(int(g[i] * SR), dtype=np.float32))
    y = np.concatenate([np.zeros(int(0.3 * SR), np.float32)] + parts + [np.zeros(int(0.4 * SR), np.float32)])
    meter = pyln.Meter(SR)
    loud = meter.integrated_loudness(y)
    y = y * (10 ** ((TARGET_LUFS - loud) / 20))
    peak = np.abs(y).max()
    if peak > 0.89:   # 約 -1 dBFS
        y = y * 0.89 / peak
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = json.loads(Path(args.pick).read_text(encoding="utf-8"))["files"] if args.pick else args.files
    y = build(files)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, y, SR, subtype="PCM_16")
    print(f"{args.out}  {len(y) / SR:.2f} s  {LAST_INFO}")


if __name__ == "__main__":
    main()
