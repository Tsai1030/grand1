"""診斷句尾：每個「語音段→停頓」的交界，量衰減時間、切點時還剩多少音量、句尾音高走向、最後一個音拖多長。

用法：python tail_diag.py 標籤=檔案或資料夾 ...
"""
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 16000
HOP = 160  # 10 ms


def ends(path):
    y, _ = librosa.load(path, sr=SR)
    rms = librosa.feature.rms(y=y, frame_length=480, hop_length=HOP)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    ref = np.percentile(db, 95)
    speech = db > ref - 40
    f0, vflag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    out = []
    i, n = 0, len(db)
    while i < n:
        if speech[i]:
            j = i
            while j < n and speech[j]:
                j += 1
            k = j
            while k < n and not speech[k]:
                k += 1
            pause = (k - j) * 0.01
            if pause >= 0.25 or j >= n:  # 句尾或片語尾
                seg = db[i:j]
                last_peak = i + int(np.argmax(seg[-40:])) + max(0, len(seg) - 40)
                peak_db = db[last_peak]
                # 從最後一個能量峰到降到 -30 dB（相對該峰）的時間
                t = last_peak
                while t < n and db[t] > peak_db - 30:
                    t += 1
                decay_ms = (t - last_peak) * 10
                # 進入停頓的瞬間，音量比該峰低多少（越小＝切得越突然）
                drop_at_cut = peak_db - (db[j] if j < n else db[-1])
                # 句尾 300 ms 有聲幀的音高變化（半音）
                v = [x for x in range(max(i, j - 30), j) if vflag[x] and not np.isnan(f0[x])]
                fall = 12 * np.log2(f0[v[-1]] / f0[v[0]]) if len(v) >= 5 else np.nan
                out.append(dict(decay_ms=decay_ms, drop_at_cut=drop_at_cut, final_st=fall,
                                last_peak_to_end_ms=(j - last_peak) * 10, pause=pause))
            i = k
        else:
            i += 1
    return out


def files_of(p):
    p = Path(p)
    return sorted(p.glob("*.wav")) if p.is_dir() else [p]


for arg in sys.argv[1:]:
    label, p = arg.split("=", 1)
    rows = [e for f in files_of(p) for e in ends(str(f))]
    if not rows:
        continue
    a = {k: np.nanmedian([r[k] for r in rows]) for k in rows[0]}
    abrupt = np.mean([r["drop_at_cut"] < 20 for r in rows])
    print(f"{label:<14} n={len(rows):3d}  衰減(ms)={a['decay_ms']:5.0f}  切點掉(dB)={a['drop_at_cut']:5.1f}  "
          f"突然收尾比例={abrupt:4.0%}  句尾音高(st)={a['final_st']:+5.2f}  最後音長(ms)={a['last_peak_to_end_ms']:4.0f}")
