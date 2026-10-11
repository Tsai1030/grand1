"""音調統計轉移：把來源語音的音高統計對齊阿公，並做保持音高的變速（Praat PSOLA，praat-parselmouth）。

stats   從 stage1/segments 算阿公的音高統計 → routeA/eval/agong_prosody_stats.json
transfer 對來源做：
  1. F0 映射（半音域的線性轉換）：st' = 阿公中位數 + (st − 來源中位數) × k，k = 阿公標準差 ÷ 來源標準差。
     中位數與起伏幅度對齊阿公；每個字的音高形狀（上升、下降、轉折）照比例保留，不會被抹平——台語聲調靠這個。
  2. 變速（只放慢、不縮短停頓）：說話中間 ≥ 0.25 秒的停頓視為句間停頓，短於 --pause-min（0.6 秒）的拉長到 0.6 秒、
     長於 --pause-max（1.0 秒）的保持原長；再算說話段的放慢倍數，讓整段語速＝--rate（預設每秒 2.65 字）。
     字數優先用 --syllables（例如祝福文的漢字數），沒給就用 rhythm.py 的自動音節數。
  音高用 Praat PSOLA（Manipulation 的 PitchTier，男聲範圍 75–300 Hz，離中位數 > 9 半音的點視為八度誤判去掉）；
  變速用 ffmpeg atempo 逐段處理（Praat 的 DurationTier 在這段 TTS 上會產生低八度假音高，所以不用）。

F0 統計一律用 librosa.pyin（60–500 Hz、16 kHz、frame 1024、hop 160，與 eval/prosody_check.py 相同）：
  median_hz、range_st（5–95 百分位差，半音）、std_st、final_fall_st（每個語句最後 200 ms 有聲段的中位數 − 該語句中位數，取平均；負值＝句尾下降）。
語句＝說話中間 ≥ 0.3 秒停頓所分開的段落。

用法：
  python prosody_transfer.py stats
  python prosody_transfer.py transfer --source in.wav --out out.wav [--syllables 98] [--rate 2.65] [--no-pitch] [--no-tempo]
  python prosody_transfer.py measure a.wav b.wav ...      # 只量 F0 統計
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STATS = os.path.join(ROOT, "routeA", "eval", "agong_prosody_stats.json")
SR = 16000
HOP = 160


def activity(y):
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP, win_length=400)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=512)
    db = 10 * np.log10(np.maximum(S[(fr >= 100) & (fr <= 3800)].sum(0), 1e-10))
    thr = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 6)
    return db, db > thr, thr


def runs_of_silence(sp, min_sec):
    idx = np.where(sp)[0]
    a, b = idx[0], idx[-1] + 1
    out, k = [], a
    while k < b:
        if not sp[k]:
            s = k
            while k < b and not sp[k]:
                k += 1
            if (k - s) * HOP / SR >= min_sec:
                out.append((s, k))
        k += 1
    return a, b, out


def f0_track(y):
    f0, v, _ = librosa.pyin(y, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=HOP)
    st = np.full(len(f0), np.nan)
    ok = v & np.isfinite(f0)
    st[ok] = 12 * np.log2(f0[ok])
    return st


def stats_of(files):
    sts, falls = [], []
    for f in files:
        y, _ = librosa.load(f, sr=SR)
        st = f0_track(y)
        _, sp, _ = activity(y)
        n = min(len(st), len(sp))
        st, sp = st[:n], sp[:n]
        sts.append(st[np.isfinite(st)])
        a, b, sil = runs_of_silence(sp, 0.3)
        bounds = [a] + [x for s, e in sil for x in (s, e)] + [b]
        for s, e in zip(bounds[0::2], bounds[1::2]):
            ph = st[s:e]
            vv = np.where(np.isfinite(ph))[0]
            if len(vv) < 20:
                continue
            last = vv[vv >= vv[-1] - 20]  # 最後 200 ms 的有聲幀
            falls.append(float(np.median(ph[last]) - np.median(ph[vv])))
    a = np.concatenate(sts)
    med = float(np.median(a))
    return {"median_hz": round(2 ** (med / 12), 1), "median_st": round(med, 3),
            "p5_hz": round(2 ** (np.percentile(a, 5) / 12), 1), "p95_hz": round(2 ** (np.percentile(a, 95) / 12), 1),
            "range_st": round(float(np.percentile(a, 95) - np.percentile(a, 5)), 2),
            "std_st": round(float(np.std(a)), 3), "final_fall_st": round(float(np.mean(falls)), 2) if falls else None,
            "n_phrases": len(falls), "voiced_frames": int(len(a))}


def count_syllables(y):
    db, sp, thr = activity(y)
    sm = np.convolve(db, np.ones(3) / 3, mode="same")
    pk, _ = find_peaks(sm, prominence=3.0, distance=10, height=thr)
    return len(pk)


def rate_info(y, nsyl=None):
    _, sp, _ = activity(y)
    a, b, sil = runs_of_silence(sp, 0.25)
    span = (b - a) * HOP / SR
    p = [(e - s) * HOP / SR for s, e in sil]
    n = nsyl or count_syllables(y)
    return {"syllables": n, "span_sec": round(span, 2), "speech_rate": round(n / span, 3),
            "pauses": len(p), "pause_mean_sec": round(float(np.mean(p)), 3) if p else 0,
            "pause_min_sec": round(min(p), 2) if p else 0, "pause_ratio": round(sum(p) / span, 3)}


def _atempo(x, sr, factor):
    import subprocess
    import tempfile
    ff = open(os.path.join(ROOT, "tools", "FFMPEG_PATH.txt"), encoding="utf-8-sig").read().strip().splitlines()[0]
    d = tempfile.mkdtemp()
    i, o = os.path.join(d, "i.wav"), os.path.join(d, "o.wav")
    sf.write(i, x, sr, subtype="FLOAT")
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", i, "-filter:a", f"atempo={factor}", "-c:a", "pcm_f32le", o], check=True)
    z, _ = sf.read(o, dtype="float64")
    return z


def stretch_chunks(z, sr, t0, t1, pauses, Lnew, f):
    """說話段用 atempo 放慢 f 倍；停頓段保留原音並在中間補靜音到新長度；頭尾靜音不動。5 ms 交叉淡化。"""
    pieces = [z[: int(t0 * sr)]]
    cur = t0
    for (ps, pe), ln in zip(pauses, Lnew):
        pieces.append(_atempo(z[int(cur * sr): int(ps * sr)], sr, 1.0 / f))
        pa = z[int(ps * sr): int(pe * sr)]
        extra = int(round(ln * sr)) - len(pa)
        if extra > 0:
            h = len(pa) // 2
            pa = np.concatenate([pa[:h], np.zeros(extra), pa[h:]])
        pieces.append(pa)
        cur = pe
    pieces.append(_atempo(z[int(cur * sr): int(t1 * sr)], sr, 1.0 / f))
    pieces.append(z[int(t1 * sr):])
    xf = int(0.005 * sr)
    out = pieces[0]
    ramp = np.linspace(1, 0, xf)
    for p in pieces[1:]:
        if len(out) >= xf and len(p) >= xf:
            out = np.concatenate([out[:-xf], out[-xf:] * ramp + p[:xf] * (1 - ramp), p[xf:]])
        else:
            out = np.concatenate([out, p])
    return out


def transfer(src, out, agong, nsyl=None, rate=2.65, pmin=0.6, pmax=1.0, do_pitch=True, do_tempo=True):
    import parselmouth
    from parselmouth.praat import call
    y, sr0 = sf.read(src, dtype="float64")
    if y.ndim > 1:
        y = y.mean(1)
    y16 = librosa.resample(y, orig_sr=sr0, target_sr=SR) if sr0 != SR else y
    s = stats_of([src])
    k = agong["std_st"] / s["std_st"]
    snd = parselmouth.Sound(y, sampling_frequency=sr0)
    man = call(snd, "To Manipulation", 0.01, 75, 300)  # 男聲範圍，減少八度誤判
    info = {"source": os.path.abspath(src), "source_stats": s, "k_range": round(k, 3),
            "shift_st": round(agong["median_st"] - s["median_st"], 2)}
    if do_pitch:
        tier = call(man, "Extract pitch tier")
        n = call(tier, "Get number of points")
        new = call("Create PitchTier", "mapped", snd.xmin, snd.xmax)
        for i in range(1, n + 1):
            t = call(tier, "Get time from index", i)
            f = call(tier, "Get value at index", i)
            if abs(12 * np.log2(f) - s["median_st"]) > 9:  # 去掉離中位數超過 9 半音的點（多半是八度誤判），由前後點內插
                continue
            st = agong["median_st"] + (12 * np.log2(f) - s["median_st"]) * k
            call(new, "Add point", t, float(2 ** (st / 12)))
        call([new, man], "Replace pitch tier")
    res = call(man, "Get resynthesis (overlap-add)")
    z = res.values[0] if do_pitch else y.copy()
    if do_tempo:
        # 變速改用 ffmpeg atempo（WSOLA）逐段處理：Praat DurationTier 在這段 TTS 上會產生大量低八度的假音高
        _, sp, _ = activity(y16)
        a, b, sil = runs_of_silence(sp, 0.25)
        pauses = [((ps * HOP / SR), (pe * HOP / SR)) for ps, pe in sil]
        L = [e - s_ for s_, e in pauses]
        Lnew = [l if l >= pmin else pmin for l in L]  # 不縮短；短於 pmin 的拉長，長於 pmax 的保持原長
        span = (b - a) * HOP / SR
        Ts = span - sum(L)
        n_syl = nsyl or count_syllables(y16)
        f = (n_syl / rate - sum(Lnew)) / Ts
        f = float(min(max(f, 1.0), 1.6))
        z = stretch_chunks(z, sr0, a * HOP / SR, b * HOP / SR, pauses, Lnew, f)
        info.update({"speech_stretch": round(f, 3), "pauses_before": [round(x, 2) for x in L],
                     "pauses_after": [round(x, 2) for x in Lnew], "syllables_used": n_syl})
    z = z / max(1.0, np.abs(z).max() / 0.95)
    sf.write(out, z, sr0, subtype="PCM_16")
    z16 = librosa.resample(z, orig_sr=sr0, target_sr=SR) if sr0 != SR else z
    info["out"] = os.path.abspath(out)
    info["out_stats"] = stats_of([out])
    info["rate_before"] = rate_info(y16, nsyl)
    info["rate_after"] = rate_info(z16, nsyl)
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    return info


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    t = sub.add_parser("transfer")
    t.add_argument("--source", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--syllables", type=int)
    t.add_argument("--syllables-from-text", help="用文字檔的漢字數當字數")
    t.add_argument("--rate", type=float, default=2.65)
    t.add_argument("--pause-min", type=float, default=0.6)
    t.add_argument("--pause-max", type=float, default=1.0)
    t.add_argument("--no-pitch", action="store_true")
    t.add_argument("--no-tempo", action="store_true")
    m = sub.add_parser("measure")
    m.add_argument("files", nargs="+")
    m.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "stats":
        files = sorted(glob.glob(os.path.join(ROOT, "stage1", "segments", "agong_*.wav")))
        st = stats_of(files)
        st.update({"source": "stage1/segments（12 段去噪片段）", "method": "librosa.pyin 60–500 Hz, 16 kHz, frame 1024, hop 160"})
        os.makedirs(os.path.dirname(STATS), exist_ok=True)
        json.dump(st, open(STATS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(st, ensure_ascii=False))
    elif a.cmd == "transfer":
        agong = json.load(open(STATS, encoding="utf-8"))
        n = a.syllables
        if a.syllables_from_text:
            n = len(re.findall(r"[一-鿿]", open(a.syllables_from_text, encoding="utf-8-sig").read()))
        info = transfer(a.source, a.out, agong, n, a.rate, a.pause_min, a.pause_max, not a.no_pitch, not a.no_tempo)
        print(json.dumps({k: info[k] for k in ("k_range", "shift_st", "source_stats", "out_stats", "rate_before", "rate_after")
                          if k in info}, ensure_ascii=False))
    else:
        rows = []
        for f in a.files:
            r = stats_of([f])
            r["file"] = os.path.basename(f)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False))
        if a.json:
            json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
