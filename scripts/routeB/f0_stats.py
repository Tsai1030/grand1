"""語調起伏：每個檔的 F0 標準差與範圍（半音）。

方法：16 kHz、librosa.pyin（fmin 60 Hz、fmax 400 Hz、hop 10 ms），只取 pyin 判為有聲的幀（voiced_flag）；
F0 換成半音（相對該檔中位數），
  f0_std_st   ＝半音標準差（越大＝起伏越大）
  f0_range_st ＝第 5 到第 95 百分位的差（半音）
  f0_median_hz
有聲幀 < 30 個的檔不算。多檔「合計」＝各檔數值的中位數。

用法：python f0_stats.py 檔案或資料夾 ... [--json out.json] [--recursive] [--chunk 秒]
--chunk N：把長檔在 ≥0.35 秒的靜音處切成約 N 秒的片段再各自算（拿來跟逐句檔比）。
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 16000
HOP = 160


def f0_of(y):
    f0, vflag, vprob = librosa.pyin(y, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    ok = vflag & np.isfinite(f0)
    return f0[ok]


def contour(y):
    """語調走向：final_st＝最後 25% 有聲幀的 F0 中位數 − 全句中位數（半音；負＝句尾往下收）；
    slope_st_s＝整句半音對時間的線性斜率（半音/秒；負＝越講越低）。"""
    f0, vflag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    ok = vflag & np.isfinite(f0)
    if ok.sum() < 30:
        return None
    t = np.arange(len(f0))[ok] * HOP / SR
    st = 12 * np.log2(f0[ok] / np.median(f0[ok]))
    n = len(st)
    final = float(np.median(st[int(n * 0.75):]))
    slope = float(np.polyfit(t, st, 1)[0]) if t[-1] - t[0] > 0.3 else 0.0
    return {"final_st": round(final, 2), "slope_st_s": round(slope, 2)}


def stats(f0):
    if len(f0) < 30:
        return None
    st = 12 * np.log2(f0 / np.median(f0))
    return {"voiced_frames": int(len(f0)), "f0_median_hz": round(float(np.median(f0)), 1),
            "f0_std_st": round(float(np.std(st)), 2),
            "f0_range_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2)}


def split_chunks(y, target):
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    thr = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 6)
    sil = db < thr
    cuts, run, last = [0], 0, 0
    for i, s in enumerate(sil):
        run = run + 1 if s else 0
        if run * HOP / SR >= 0.35 and (i * HOP - last) / SR >= target:
            c = (i - run // 2) * HOP
            cuts.append(c)
            last = c
            run = 0
    cuts.append(len(y))
    return [y[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if (b - a) / SR > 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--chunk", type=float, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    files = []
    for p in args.inputs:
        p = Path(p)
        files += sorted(p.rglob("*.wav") if args.recursive else p.glob("*.wav")) if p.is_dir() else [p]
    rows = []
    for f in files:
        y, _ = librosa.load(str(f), sr=SR)
        parts = split_chunks(y, args.chunk) if args.chunk > 0 else [y]
        for k, part in enumerate(parts):
            s = stats(f0_of(part))
            if not s:
                continue
            name = str(f) if len(parts) == 1 else f"{f}#{k}"
            rows.append({"file": name, "sec": round(len(part) / SR, 2), **s})
            print(f"{name}\t{rows[-1]['sec']}\t{s['f0_median_hz']}\t{s['f0_std_st']}\t{s['f0_range_st']}", flush=True)
    summ = {k: round(float(np.median([r[k] for r in rows])), 2) for k in ["f0_std_st", "f0_range_st", "f0_median_hz"]} if rows else {}
    print("中位數", summ)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"method": __doc__.split("用法")[0].strip(), "rows": rows, "median": summ},
                                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
