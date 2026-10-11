"""聲調風險初篩：轉換後語音的 F0 輪廓跟台語原音比相關係數。

做法（每個檔都一樣）：
  1. 16 kHz、hop 10 ms，用 librosa.pyin 抓 F0（fmin 60 Hz、fmax 400 Hz）。
  2. 換成半音，減掉該檔有聲段的中位數（去掉男女/不同人的音高差，只留起伏）。
  3. 用 MFCC（每檔各自做均值/方差正規化）做 DTW，把輸出對齊回原音的時間軸
     （v2 convert-style 會改長度，所以一律用 DTW；v1 與 style 關閉時 DTW 幾乎是對角線）。
  4. 在兩邊都有聲的幀上算：
     - r_global：整段半音輪廓的 Pearson 相關
     - r_local ：減掉 0.5 秒移動中位數後（去掉句調，只留字調等級的起伏）的 Pearson 相關
     - rmse_st ：半音誤差均方根（各自去中位數後）
     - vuv_agree：有聲/無聲判斷一致的比例
  r 越高代表音高走向越貼近原音，聲調被改壞的風險越低。這只是初篩，不能取代會台語的人逐句聽。

用法：
  python f0_corr.py --ref 原音.wav --hyp 輸出1.wav [輸出2.wav ...] [--json out.json]
"""
import argparse
import json
import os

import numpy as np
import librosa
from scipy.ndimage import median_filter

SR = 16000
HOP = 160
FMIN, FMAX = 60.0, 400.0
_cache = {}


def analyze(path):
    if path in _cache:
        return _cache[path]
    y, _ = librosa.load(path, sr=SR)
    f0, vflag, vprob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=1024, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=512, hop_length=HOP)
    n = min(len(f0), mfcc.shape[1])
    f0, vflag, mfcc = f0[:n], vflag[:n], mfcc[:, :n]
    mfcc = (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-8)
    st = np.full(n, np.nan)
    v = vflag & np.isfinite(f0)
    if v.sum() > 0:
        st[v] = 12 * np.log2(f0[v])
        st[v] -= np.median(st[v])
    med_hz = float(np.nanmedian(f0[v])) if v.sum() else float("nan")
    _cache[path] = (st, v, mfcc, med_hz, len(y) / SR)
    return _cache[path]


def detrend(st, v, win_frames=50):
    out = np.full_like(st, np.nan)
    if v.sum() < 5:
        return out
    filled = st.copy()
    idx = np.arange(len(st))
    filled[~v] = np.interp(idx[~v], idx[v], st[v])
    trend = median_filter(filled, size=win_frames, mode="nearest")
    out[v] = st[v] - trend[v]
    return out


def pearson(a, b):
    if len(a) < 10 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(ref, hyp):
    st_r, v_r, m_r, hz_r, dur_r = analyze(ref)
    st_h, v_h, m_h, hz_h, dur_h = analyze(hyp)
    _, wp = librosa.sequence.dtw(X=m_r, Y=m_h, metric="euclidean")
    wp = wp[::-1]
    # 每個原音幀取第一個對到的輸出幀
    map_h = np.full(len(st_r), -1)
    for i, j in wp:
        if map_h[i] < 0:
            map_h[i] = j
    ok = map_h >= 0
    ri = np.where(ok)[0]
    hj = map_h[ok]
    dl_r = detrend(st_r, v_r)
    dl_h = detrend(st_h, v_h)
    both = v_r[ri] & v_h[hj]
    a, b = st_r[ri][both], st_h[hj][both]
    la, lb = dl_r[ri][both], dl_h[hj][both]
    return {
        "hyp": os.path.basename(hyp),
        "r_global": round(pearson(a, b), 3),
        "r_local": round(pearson(la, lb), 3),
        "rmse_st": round(float(np.sqrt(np.mean((a - b) ** 2))), 2) if len(a) else None,
        "vuv_agree": round(float(np.mean(v_r[ri] == v_h[hj])), 3),
        "n_voiced_pairs": int(both.sum()),
        "ref_median_hz": round(hz_r, 1),
        "hyp_median_hz": round(hz_h, 1),
        "ref_sec": round(dur_r, 2),
        "hyp_sec": round(dur_h, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", nargs="+", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = [compare(args.ref, h) for h in args.hyp]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"ref": os.path.abspath(args.ref), "method": "pyin 60-400Hz hop10ms, semitone-median, MFCC-DTW", "rows": rows},
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
