"""N 系列：原速 OmniVoice 逐句檔重挑 seed，用加停頓放慢（不拉長字）。用法：python tail_fix2.py prep"""
import json, sys
from pathlib import Path
import numpy as np, librosa, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
import concat, tail_fix as tf

ROOT, W, SR = tf.ROOT, tf.W, 24000
HOP = tf.HOP
COMMA = {2: [0.35], 4: [0.35], 5: [0.3, 0.3], 6: [0.35], 7: [0.35]}


def pick():
    rows = json.load(open(ROOT / "routeB/eval/candidates.json", encoding="utf-8"))["methods"]["ov_design_design"]["rows"]
    out = {}
    for sid in range(1, 9):
        pool = [r for r in rows if r["id"] == sid]
        ok = [r for r in pool if not r["asr_problem"]] or pool
        best = max(ok, key=lambda r: r["tone_r"] - 0.04 * abs(r["final_st"] + 1.0))
        out[sid] = best
    return out


def split_pauses(y, n_cut, pauses):
    db = tf.edb(y)
    n = len(db)
    thr = max(np.percentile(db, 95) - 35, np.percentile(db, 5) + 6)
    sil = db < thr
    runs, s = [], None
    for i in range(n):
        if sil[i] and s is None: s = i
        if (not sil[i]) and s is not None:
            runs.append((s, i)); s = None
    # 只看句中（不含頭尾），且至少 60 ms
    inner = [r for r in runs if r[0] > 0.15 * n and r[1] < 0.9 * n and (r[1] - r[0]) >= 6 and r[1] < n]
    inner = sorted(sorted(inner, key=lambda r: r[0] - r[1])[:n_cut])
    parts, prev = [], 0
    for (a, b), p in zip(inner, pauses):
        seg = y[prev:a * HOP + int(0.15 * SR)] if False else y[prev:a * HOP + int(0.06 * SR)]
        parts.append(tail_fade(seg)); parts.append(np.zeros(int(p * SR), np.float32))
        prev = max(a * HOP, b * HOP - int(0.03 * SR))
    parts.append(y[prev:])
    return np.concatenate(parts), len(inner)


def tail_fade(y, ms=60):
    y = y.copy(); m = min(int(SR * ms / 1000), len(y))
    y[-m:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, m)))
    return y


def fade(y, sr, ms=10):
    y = y.copy(); n = int(sr * 0.01)
    y[:n] *= np.linspace(0, 1, n)
    m = int(sr * 0.07)
    y[-m:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, m)))
    return y


def prep():
    P = pick()
    print({k: (v["seed"], v["tone_r"], v["final_st"]) for k, v in P.items()})
    d = W / "N1_sent"; d.mkdir(exist_ok=True)
    outs = []
    for sid, r in P.items():
        y, _ = librosa.load(str(ROOT / r["file"]), sr=SR)
        y, _ = tf.trim_keep(y, tail=0.15)
        if sid in COMMA:
            y, k = split_pauses(y, len(COMMA[sid]), COMMA[sid])
            print("sent", sid, "cuts", k)
        sf.write(d / f"s{sid}.wav", y, SR); outs.append(d / f"s{sid}.wav")
    concat.trim = lambda y, sr, margin=0.04: y
    concat.fade = fade
    concat.GAP_MIN, concat.GAP_MAX = 0.8, 1.3
    y = concat.build(outs, target_rate=2.75)
    sf.write(W / "N1_src.wav", y, SR, subtype="PCM_16")
    print(len(y) / SR, concat.LAST_INFO)


if __name__ == "__main__":
    prep()
