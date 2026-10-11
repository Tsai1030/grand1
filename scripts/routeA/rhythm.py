"""節奏量測：語速（每秒音節數）、構音速率、停頓比例。

台語一字一音節，所以用「音節核」數估音節數（de Jong & Wempe 2009 的簡化版）：
  - 16 kHz，強度＝100–3800 Hz 頻帶能量（dB；STFT 25 ms 視窗、hop 10 ms）；
  - 說話／停頓：強度高於「全檔第 95 百分位 − 30 dB」且高於「第 10 百分位 + 6 dB」的幀算說話；
    低於門檻、連續 ≥ 200 ms 的段落算停頓（句首句尾的靜音不算，只算說話中間的停頓）；
  - 音節核：3 幀平滑後的強度峰（突出度 ≥ 3 dB、峰間 ≥ 100 ms、高於說話門檻）。
    不要求 pyin 判有聲：來源錄音有不少明顯在講話的幀 pyin 判不出有聲，加這條件會少算一半。
    參數敏感度：突出度 2–4 dB、平滑 3–5 幀之間，阿公／來源的語速比在 1.08–1.27 之間。
指標：
  speech_rate        音節數 ÷（第一個到最後一個說話幀的時間）       ← 聽起來的「快慢」
  articulation_rate  音節數 ÷（扣掉中間停頓後的說話時間）            ← 字本身念得多快
  pause_ratio        中間停頓總長 ÷（第一個到最後一個說話幀的時間）
  pauses_per_10s、mean_pause_sec
多檔「合計」＝總音節數／總時間（每檔各自去掉頭尾靜音）。
自動數音節有誤差（連音、弱讀會少算），只用來比「同一把尺量出來的相對快慢」。

用法：python rhythm.py 檔案或資料夾 ... [--json out.json] [--label 名稱]
"""
import argparse
import glob
import json
import os

import numpy as np
import librosa
from scipy.signal import find_peaks

SR = 16000
HOP = 160
WIN = 800  # 50 ms


def analyze(path, prom=3.0, smooth=3):
    y, _ = librosa.load(path, sr=SR)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP, win_length=400)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=512)
    band = (fr >= 100) & (fr <= 3800)   # 只看 100–3800 Hz：阿公錄音 4 kHz 以上幾乎沒能量，來源有，避免高頻雜音影響
    db = 10 * np.log10(np.maximum(S[band].sum(0), 1e-10))
    thr = max(np.percentile(db, 95) - 30.0, np.percentile(db, 10) + 6.0)
    speech = db > thr
    idx = np.where(speech)[0]
    if len(idx) == 0:
        return None
    a, b = idx[0], idx[-1] + 1
    span = (b - a) * HOP / SR
    # 中間停頓
    pauses = []
    run = 0
    for k in range(a, b):
        if not speech[k]:
            run += 1
        else:
            if run * HOP / SR >= 0.2:
                pauses.append(run * HOP / SR)
            run = 0
    pause_total = float(sum(pauses))
    sm = np.convolve(db, np.ones(smooth) / smooth, mode="same")
    peaks, _ = find_peaks(sm, prominence=prom, distance=10, height=thr)
    nsyl = len(peaks)
    art_time = max(span - pause_total, 1e-6)
    return {
        "file": os.path.basename(path),
        "duration_sec": round(len(y) / SR, 2),
        "span_sec": round(span, 2),
        "syllables": nsyl,
        "speech_rate": round(nsyl / span, 3),
        "articulation_rate": round(nsyl / art_time, 3),
        "pause_total_sec": round(pause_total, 2),
        "pause_ratio": round(pause_total / span, 3),
        "pauses": len(pauses),
        "pauses_per_10s": round(len(pauses) / span * 10, 2),
        "mean_pause_sec": round(pause_total / len(pauses), 3) if pauses else 0.0,
    }


def pooled(rows):
    syl = sum(r["syllables"] for r in rows)
    span = sum(r["span_sec"] for r in rows)
    pt = sum(r["pause_total_sec"] for r in rows)
    npz = sum(r["pauses"] for r in rows)
    return {"files": len(rows), "syllables": syl, "span_sec": round(span, 2),
            "speech_rate": round(syl / span, 3), "articulation_rate": round(syl / (span - pt), 3),
            "pause_ratio": round(pt / span, 3), "pauses_per_10s": round(npz / span * 10, 2),
            "mean_pause_sec": round(pt / npz, 3) if npz else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    files = []
    for p in args.inputs:
        files += sorted(glob.glob(os.path.join(p, "*.wav"))) if os.path.isdir(p) else [p]
    rows = [r for r in (analyze(f) for f in files) if r]
    print("file\tsec\tsyl\tspeech_rate\tartic_rate\tpause_ratio\tpauses/10s\tmean_pause")
    for r in rows:
        print(f"{r['file']}\t{r['duration_sec']}\t{r['syllables']}\t{r['speech_rate']}\t{r['articulation_rate']}\t"
              f"{r['pause_ratio']}\t{r['pauses_per_10s']}\t{r['mean_pause_sec']}")
    pl = pooled(rows)
    print(f"合計({pl['files']})\t{pl['span_sec']}\t{pl['syllables']}\t{pl['speech_rate']}\t{pl['articulation_rate']}\t"
          f"{pl['pause_ratio']}\t{pl['pauses_per_10s']}\t{pl['mean_pause_sec']}")
    if args.json:
        json.dump({"label": args.label, "method": __doc__.split("用法")[0].strip(), "files": rows, "pooled": pl},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
