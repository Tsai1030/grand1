"""產生轉換前的來源／參考音變體（都不改音高）。

  --clean-ref   從 stage1/segments 挑聲紋最一致的幾段（預設 agong_06、08、07），中間補 0.3 秒靜音串成 20–25 秒參考音
  --atempo F    用 ffmpeg atempo 把來源整段變速 F 倍（保持音高）
  --pause-cap S 只縮短停頓：說話中間 ≥ 0.25 秒的停頓，超過 S 秒的部分從停頓中間剪掉（10 ms 交叉淡化），字本身不變速

用法：
  python make_variants.py --clean-ref --out routeA/sources/ref_clean_06-08-07.wav
  python make_variants.py --source inputs/taiwanese_source_joined.wav --atempo 1.16 --out routeA/sources/src_atempo1.16.wav
  python make_variants.py --source inputs/taiwanese_source_joined.wav --pause-cap 0.45 --out routeA/sources/src_pausecap0.45.wav
"""
import argparse
import os
import subprocess

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def clean_ref(ids, out, gap=0.3):
    parts = []
    sr0 = None
    for i in ids:
        y, sr = sf.read(os.path.join(ROOT, "stage1", "segments", f"agong_{i}.wav"), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        sr0 = sr0 or sr
        assert sr == sr0
        parts += [y, np.zeros(int(gap * sr), dtype=np.float32)]
    y = np.concatenate(parts[:-1])
    sf.write(out, y, sr0, subtype="PCM_16")
    print(f"clean ref {ids}: {len(y)/sr0:.2f}s -> {out}")


def atempo(src, f, out):
    ff = open(os.path.join(ROOT, "tools", "FFMPEG_PATH.txt"), encoding="utf-8-sig").read().strip().splitlines()[0]
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-filter:a", f"atempo={f}",
                    "-ac", "1", "-c:a", "pcm_s16le", out], check=True)
    d = sf.info(out)
    print(f"atempo {f}: {d.frames/d.samplerate:.2f}s -> {out}")


def pause_cap(src, cap, out, min_pause=0.25, xfade=0.01):
    y, sr = sf.read(src, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    import librosa
    hop = int(sr * 0.01)
    S = np.abs(librosa.stft(y, n_fft=int(sr * 0.032), hop_length=hop, win_length=int(sr * 0.025))) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=int(sr * 0.032))
    band = (fr >= 100) & (fr <= 3800)
    db = 10 * np.log10(np.maximum(S[band].sum(0), 1e-10))
    thr = max(np.percentile(db, 95) - 30.0, np.percentile(db, 10) + 6.0)
    speech = db > thr
    idx = np.where(speech)[0]
    a, b = idx[0], idx[-1]
    cuts = []  # (start_sample, end_sample) to remove
    k = a
    while k <= b:
        if not speech[k]:
            s = k
            while k <= b and not speech[k]:
                k += 1
            L = (k - s) * 0.01
            if L >= min_pause and L > cap:
                remove = L - cap
                mid = (s + k) / 2 * hop
                cuts.append((int(mid - remove * sr / 2), int(mid + remove * sr / 2)))
        k += 1
    xf = int(xfade * sr)
    pieces = []
    prev = 0
    for (c0, c1) in cuts:
        pieces.append(y[prev:c0])
        prev = c1
    pieces.append(y[prev:])
    outy = pieces[0]
    fade_out = np.linspace(1, 0, xf, dtype=np.float32)
    for p in pieces[1:]:
        if len(outy) >= xf and len(p) >= xf:
            mix = outy[-xf:] * fade_out + p[:xf] * (1 - fade_out)
            outy = np.concatenate([outy[:-xf], mix, p[xf:]])
        else:
            outy = np.concatenate([outy, p])
    sf.write(out, outy, sr, subtype="PCM_16")
    removed = sum(c1 - c0 for c0, c1 in cuts) / sr
    print(f"pause-cap {cap}s: {len(cuts)} pauses shortened, removed {removed:.2f}s, {len(y)/sr:.2f}s -> {len(outy)/sr:.2f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source")
    ap.add_argument("--clean-ref", action="store_true")
    ap.add_argument("--ids", nargs="*", default=["06", "08", "07"])
    ap.add_argument("--atempo", type=float)
    ap.add_argument("--pause-cap", type=float)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.clean_ref:
        clean_ref(a.ids, a.out)
    elif a.atempo:
        atempo(a.source, a.atempo, a.out)
    elif a.pause_cap:
        pause_cap(a.source, a.pause_cap, a.out)


if __name__ == "__main__":
    main()
