"""產生 routeA/aug_data/：阿公錄音的增強資料集（目標總長 8–10 分鐘）。

(a) windows/ 重新切窗：從 stage1/enhanced/male_voice_only_denoise.wav 的 0–96.07 秒（排除 agong_13 之後）
    找停頓（100–3800 Hz 能量低於門檻、≥ 0.15 秒），以停頓中點當切點，隨機挑 3–12 秒、起點錯開、可重疊的視窗。
(b) perturb/ 輕微擾動（套在 stage1/segments 的 12 段上，每類各挑一部分）：
    pitch  音高 ±0.5 半音（Praat PSOLA，不變速）
    tempo  保持音高的變速 0.9／1.1（ffmpeg atempo）
    reverb 小房間殘響（合成脈衝響應 RT60 約 0.25 秒，濕聲 25%）
    gain   音量 −6／+4 dB（不做響度正規化，峰值限制在 −1 dBFS）
除 gain 外每段都正規化到 −20 LUFS（與 stage1 相同）。輸出 44.1 kHz 16-bit 單聲道。固定種子 1234。
用法：python make_aug_data.py
"""
import glob
import json
import os
import random
import subprocess

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "routeA", "aug_data")
SRC = os.path.join(ROOT, "stage1", "enhanced", "male_voice_only_denoise.wav")
SEGS = sorted(glob.glob(os.path.join(ROOT, "stage1", "segments", "agong_*.wav")))
END_SEC = 96.07
rng = random.Random(1234)
nrng = np.random.default_rng(1234)


def norm(y, sr, target=-20.0):
    m = pyln.Meter(sr)
    l = m.integrated_loudness(y)
    y = pyln.normalize.loudness(y, l, target) if np.isfinite(l) else y
    pk = np.abs(y).max()
    return y * (0.89 / pk) if pk > 0.89 else y


def write(path, y, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), sr, subtype="PCM_16")
    return len(y) / sr


def pauses(y, sr):
    hop = int(sr * 0.01)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop, win_length=int(sr * 0.025))) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    db = 10 * np.log10(np.maximum(S[(fr >= 100) & (fr <= 3800)].sum(0), 1e-10))
    thr = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 6)
    sil = db <= thr
    cuts, k, n = [0.0], 0, len(sil)
    while k < n:
        if sil[k]:
            s = k
            while k < n and sil[k]:
                k += 1
            if (k - s) * 0.01 >= 0.15:
                cuts.append((s + k) / 2 * 0.01)
        k += 1
    return sorted(set(c for c in cuts if c <= len(y) / sr))


def make_windows(target_n=40):
    y, sr = sf.read(SRC, dtype="float32")
    y = y[: int(END_SEC * sr)]
    cuts = pauses(y, sr)
    cand = [(a, b) for i, a in enumerate(cuts) for b in cuts[i + 1:] if 3.0 <= b - a <= 12.0]
    rng.shuffle(cand)
    chosen, starts = [], []
    for a, b in cand:  # 起點錯開：跟已選視窗的起點至少差 0.8 秒；同一對切點不重複
        if all(abs(a - s) >= 0.8 or abs(b - e) >= 0.8 for s, e in chosen):
            chosen.append((a, b))
        if len(chosen) >= target_n:
            break
    chosen.sort()
    rows = []
    for i, (a, b) in enumerate(chosen):
        seg = norm(y[int(a * sr): int(b * sr)].astype(np.float64), sr)
        d = write(os.path.join(OUT, "windows", f"win_{i:02d}_{a:06.2f}-{b:06.2f}.wav"), seg, sr)
        rows.append({"file": f"windows/win_{i:02d}_{a:06.2f}-{b:06.2f}.wav", "start": round(a, 2), "end": round(b, 2), "sec": round(d, 2)})
    return rows, len(cuts)


def pitch_shift(y, sr, st):
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    man = call(snd, "To Manipulation", 0.01, 60, 400)
    tier = call(man, "Extract pitch tier")
    call(tier, "Multiply frequencies", snd.xmin, snd.xmax, 2 ** (st / 12))
    call([tier, man], "Replace pitch tier")
    return call(man, "Get resynthesis (overlap-add)").values[0]


def atempo(path, f, out):
    ff = open(os.path.join(ROOT, "tools", "FFMPEG_PATH.txt"), encoding="utf-8-sig").read().strip().splitlines()[0]
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-filter:a", f"atempo={f}", "-ac", "1",
                    "-c:a", "pcm_s16le", out], check=True)
    y, sr = sf.read(out, dtype="float32")
    return y, sr


def reverb(y, sr, rt60=0.25, wet=0.25):
    n = int(sr * rt60 * 1.5)
    t = np.arange(n) / sr
    ir = nrng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[: int(sr * 0.003)] = 0
    ir /= np.sqrt((ir ** 2).sum())
    wetsig = np.convolve(y, ir)[: len(y)]
    return (1 - wet) * y + wet * wetsig


def make_perturb():
    rows = []
    idx = list(range(len(SEGS)))
    for i in idx:  # 音高：12 段，正負交替
        y, sr = sf.read(SEGS[i], dtype="float64")
        st = 0.5 if i % 2 == 0 else -0.5
        name = f"perturb/pitch{st:+.1f}_{os.path.basename(SEGS[i])}"
        d = write(os.path.join(OUT, name), norm(pitch_shift(y, sr, st), sr), sr)
        rows.append({"file": name, "type": "pitch", "param": st, "sec": round(d, 2)})
    tmp = os.path.join(OUT, "_tmp.wav")
    for i in idx:  # 變速：12 段，0.9／1.1 交替
        f = 0.9 if i % 2 == 0 else 1.1
        y, sr = atempo(SEGS[i], f, tmp)
        name = f"perturb/tempo{f}_{os.path.basename(SEGS[i])}"
        d = write(os.path.join(OUT, name), norm(y.astype(np.float64), sr), sr)
        rows.append({"file": name, "type": "tempo", "param": f, "sec": round(d, 2)})
    os.remove(tmp)
    sel = rng.sample(idx, 6)
    for i in sel:  # 殘響：6 段
        y, sr = sf.read(SEGS[i], dtype="float64")
        name = f"perturb/reverb_{os.path.basename(SEGS[i])}"
        d = write(os.path.join(OUT, name), norm(reverb(y, sr), sr), sr)
        rows.append({"file": name, "type": "reverb", "param": "rt60 0.25s wet 0.25", "sec": round(d, 2)})
    rest = [i for i in idx if i not in sel]
    for k, i in enumerate(rest):  # 音量：其餘 6 段
        y, sr = sf.read(SEGS[i], dtype="float64")
        g = -6.0 if k % 2 == 0 else 4.0
        z = y * 10 ** (g / 20)
        pk = np.abs(z).max()
        z = z * (0.89 / pk) if pk > 0.89 else z
        name = f"perturb/gain{g:+.0f}dB_{os.path.basename(SEGS[i])}"
        d = write(os.path.join(OUT, name), z, sr)
        rows.append({"file": name, "type": "gain", "param": g, "sec": round(d, 2)})
    return rows


def main():
    win, ncuts = make_windows()
    per = make_perturb()
    summ = {"windows": {"n": len(win), "sec": round(sum(r["sec"] for r in win), 1)}}
    for t in ("pitch", "tempo", "reverb", "gain"):
        rr = [r for r in per if r["type"] == t]
        summ[t] = {"n": len(rr), "sec": round(sum(r["sec"] for r in rr), 1)}
    summ["total"] = {"n": len(win) + len(per), "sec": round(sum(r["sec"] for r in win + per), 1)}
    json.dump({"summary": summ, "pause_cut_points": ncuts, "windows": win, "perturb": per},
              open(os.path.join(OUT, "aug_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
