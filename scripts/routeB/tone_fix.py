"""聲調修正：V1/V2/V3 挑 seed 串接，V4 音高移植。用法：python tone_fix.py build"""
import json, sys
from pathlib import Path
import numpy as np, librosa, soundfile as sf, parselmouth
from parselmouth.praat import call
from scipy.signal import medfilt
sys.path.insert(0, str(Path(__file__).parent))
import concat, tail_fix as tf, tail_fix2 as t2, tone_dtw as td

ROOT, W, SR = tf.ROOT, tf.W, 24000
S = json.load(open(W / "tone_scores.json", encoding="utf-8"))


def best(versions, sid):
    c = [x for x in S if x["v"] in versions and x["sid"] == sid and x["ok"] and x["tone"] is not None]
    return max(c, key=lambda x: x["tone"])


def transplant(y, sid, std_target=3.2):
    """y: 24k 已 trim 的 OmniVoice 句；從網頁 TTS 對應句移植 F0 走向。"""
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    fa, sa = td._feat(y16)
    fb, sb = td._src_feat(sid)
    _, wp = librosa.sequence.dtw(X=fa, Y=fb, metric="euclidean")
    wp = wp[::-1]
    tgt = np.full(fa.shape[1], np.nan)
    for k in range(fa.shape[1]):
        v = sb[wp[wp[:, 0] == k, 1]]
        v = v[np.isfinite(v)]
        if len(v): tgt[k] = np.median(v)
    gstd = np.nanstd(np.concatenate([td._src_feat(i)[1] for i in range(1, 9)]))
    tgt = tgt * (std_target / gstd)
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    man = call(snd, "To Manipulation", 0.01, 75, 400)
    pt = call(man, "Extract pitch tier")
    n = call(pt, "Get number of points")
    ts = [call(pt, "Get time from index", i + 1) for i in range(n)]
    fs = np.array([call(pt, "Get value at index", i + 1) for i in range(n)])
    med = np.median(fs)
    new = call("Create PitchTier", "p", 0, snd.xmax)
    used = 0
    for t, f in zip(ts, fs):
        k = int(round(t / 0.01))
        if 0 <= k < len(tgt) and np.isfinite(tgt[k]) and sa[k] == sa[k]:
            f = med * 2 ** (tgt[k] / 12); used += 1
        call(new, "Add point", t, float(f))
    call([man, new], "Replace pitch tier")
    out = call(man, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    return out, used / max(n, 1)


def build_pass(name, files, std_pairs=None):
    d = W / f"{name}_sent"; d.mkdir(exist_ok=True)
    outs = []
    for sid, f in enumerate(files, 1):
        y, _ = librosa.load(str(f), sr=SR)
        y, _ = tf.trim_keep(y, tail=0.15)
        if isinstance(std_pairs, dict):
            y, frac = transplant(y, sid); print(name, sid, "移植比例", round(frac, 2))
        if sid in t2.COMMA:
            y, k = t2.split_pauses(y, len(t2.COMMA[sid]), t2.COMMA[sid])
        sf.write(d / f"s{sid}.wav", y, SR); outs.append(d / f"s{sid}.wav")
    concat.trim = lambda y, sr, margin=0.04: y
    concat.fade = t2.fade
    concat.GAP_MIN, concat.GAP_MAX = 0.8, 1.3
    y = concat.build(outs, target_rate=2.75)
    sf.write(W / f"{name}_src.wav", y, SR, subtype="PCM_16")
    print(name, len(y) / SR, concat.LAST_INFO)


def main():
    v1 = [best(["tone_v1"], i) for i in range(1, 9)]
    v2 = [best(["tone_v2"], i) for i in range(1, 9)]
    v3 = [best(["tone_v1", "tone_v2"], i) for i in range(1, 9)]
    for n, p in (("V1", v1), ("V2", v2), ("V3", v3)):
        print(n, [(x["v"][-1], x["seed"], x["tone"]) for x in p], "mean", round(np.mean([x["tone"] for x in p]), 3))
        build_pass(n, [ROOT / x["file"] for x in p])
    n1 = t2.pick()
    build_pass("V4", [ROOT / n1[i]["file"] for i in range(1, 9)], std_pairs={})


if __name__ == "__main__":
    main()
