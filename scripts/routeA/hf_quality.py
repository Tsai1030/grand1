"""高頻乾淨度（金屬／電子質感的粗略客觀指標）。

全部重取樣到 22.05 kHz（輸出本身就是 22.05 kHz；來源只有 16 kHz，8 kHz 以上本來就是空的），
STFT n_fft 1024、hop 256，只看「說話幀」（全頻能量高於第 95 百分位 − 30 dB）：
  hf_ratio_db   4–8 kHz 能量 ÷ 0–4 kHz 能量（dB）。越高＝高頻越多（嘶聲、電子聲也會讓它變高）。
  hf_flatness   4–8 kHz 的頻譜平坦度（幾何平均÷算術平均，0–1）逐幀取中位數。
                金屬／嗡嗡的電子聲＝高頻有一條條諧波（像音調）→ 平坦度低；自然的氣音、齒音較像雜訊 → 平坦度高。
  hf_peakiness  4–8 kHz 每幀「最大值 ÷ 中位數」（dB）的中位數：越高＝高頻有尖銳的單頻峰（金屬感、鈴聲）。
這三個數只能當「同一段原音、同一把尺」下的相對比較，最後仍以耳朵為準。

用法：python hf_quality.py 檔案或資料夾 ... [--json out.json]
"""
import argparse
import glob
import json
import os

import numpy as np
import librosa

SR = 22050


def analyze(path):
    y, _ = librosa.load(path, sr=SR)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2 + 1e-12
    fr = librosa.fft_frequencies(sr=SR, n_fft=1024)
    tot = 10 * np.log10(S.sum(0))
    sp = tot > np.percentile(tot, 95) - 30
    lo = (fr > 50) & (fr < 4000)
    hi = (fr >= 4000) & (fr < 8000)
    Sh = S[hi][:, sp]
    Sl = S[lo][:, sp]
    ratio = 10 * np.log10(Sh.sum() / Sl.sum())
    flat = np.exp(np.mean(np.log(Sh), axis=0)) / np.mean(Sh, axis=0)
    peak = 10 * np.log10(Sh.max(0) / np.median(Sh, axis=0))
    return {"file": os.path.basename(path), "hf_ratio_db": round(float(ratio), 2),
            "hf_flatness": round(float(np.median(flat)), 4), "hf_peakiness_db": round(float(np.median(peak)), 2),
            "speech_frames": int(sp.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    files = []
    for p in a.inputs:
        files += sorted(glob.glob(os.path.join(p, "*.wav"))) if os.path.isdir(p) else [p]
    rows = [analyze(f) for f in files]
    print("file\thf_ratio_db\thf_flatness\thf_peakiness_db")
    for r in rows:
        print(f"{r['file']}\t{r['hf_ratio_db']}\t{r['hf_flatness']}\t{r['hf_peakiness_db']}")
    if a.json:
        json.dump({"method": __doc__.split("用法")[0].strip(), "rows": rows}, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
