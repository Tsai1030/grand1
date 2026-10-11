"""口音／聲調指標：每句 F0 走向跟「網頁 TTS 原音」（台灣台語腔）同一句的相關。

原音 inputs/taiwanese_source_joined.wav 唸的是舊稿（含「大家好」），共 15 個短語、14 個停頓
（7 個逗號＋7 個句號）。照 rhythm.py 的說話／停頓判斷找出 ≥0.2 秒的停頓，取最長的 14 個切成 15 段，
對到 8 句：s1＝第 2 段（我是阿公）、s2＝3–4、s3＝5、s4＝6–7、s5＝8–10、s6＝11–12、s7＝13–14、s8＝15。
相關的算法：兩邊各取 MFCC（13 維＋一階差分），用 DTW 對齊（內容對齊，不看音高），
再沿對齊路徑取兩邊都有聲的幀，F0 換成半音（各自減掉中位數），算 Pearson 相關 tone_r。
tone_r 越高＝每個字的音高走向越接近台灣腔原音（聲調沒走樣）。原音本身也不完美，只當參考。
用法：python tone_dtw.py 檔案（s<句號>_*.wav）...
"""
import re
import sys
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "taiwanese_source_joined.wav"
SR = 16000
HOP = 160
MAP = {1: [1], 2: [2, 3], 3: [4], 4: [5, 6], 5: [7, 8, 9], 6: [10, 11], 7: [12, 13], 8: [14]}  # 0 起算的段號


def _speech_mask(y):
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP, win_length=400)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=512)
    band = (fr >= 100) & (fr <= 3800)
    db = 10 * np.log10(np.maximum(S[band].sum(0), 1e-10))
    thr = max(np.percentile(db, 95) - 30.0, np.percentile(db, 10) + 6.0)
    return db > thr


@lru_cache(maxsize=1)
def source_sentences():
    y, _ = librosa.load(str(SRC), sr=SR)
    sp = _speech_mask(y)
    idx = np.where(sp)[0]
    a, b = idx[0], idx[-1] + 1
    pauses, run = [], 0
    for k in range(a, b):
        if not sp[k]:
            run += 1
        else:
            if run * HOP / SR >= 0.2:
                pauses.append((run, k - run, k))
            run = 0
    pauses = sorted(sorted(pauses, reverse=True)[:14], key=lambda p: p[1])
    cuts = [a * HOP] + [((s + e) // 2) * HOP for _, s, e in pauses] + [b * HOP]
    segs = [y[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
    if len(segs) != 15:
        raise RuntimeError(f"原音切出 {len(segs)} 段（預期 15）")
    return {sid: np.concatenate([segs[i] for i in ids]) for sid, ids in MAP.items()}


def _feat(y):
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=HOP, n_fft=512)
    m = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)
    f = np.vstack([m, librosa.feature.delta(m)])
    f0, vflag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    n = min(f.shape[1], len(f0))
    f0, vflag, f = f0[:n], vflag[:n], f[:, :n]
    st = np.full(n, np.nan)
    ok = vflag & np.isfinite(f0)
    if ok.sum() > 10:
        st[ok] = 12 * np.log2(f0[ok] / np.median(f0[ok]))
    return f, st


@lru_cache(maxsize=16)
def _src_feat(sid):
    return _feat(source_sentences()[sid])


def tone_r(y16, sid):
    fa, sa = _feat(y16)
    fb, sb = _src_feat(sid)
    _, wp = librosa.sequence.dtw(X=fa, Y=fb, metric="euclidean")
    wp = wp[::-1]
    va, vb = sa[wp[:, 0]], sb[wp[:, 1]]
    ok = np.isfinite(va) & np.isfinite(vb)
    if ok.sum() < 20:
        return None
    return round(float(np.corrcoef(va[ok], vb[ok])[0, 1]), 3)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    src = source_sentences()
    print({k: round(len(v) / SR, 2) for k, v in src.items()})
    for f in sys.argv[1:]:
        sid = int(re.match(r"s(\d+)_", Path(f).name).group(1))
        y, _ = librosa.load(f, sr=SR)
        print(f, tone_r(y, sid))
