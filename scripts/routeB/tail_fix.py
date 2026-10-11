"""尾音修正：T1 延長最後一個音＋長淡出；T2 再放緩句尾音高；T3 後製底噪；T4 逐句 VC。
用法：python tail_fix.py prep   -> routeB/tail_work/{T1,T2}_sent/*.wav + 串好的 T1/T2 來源
      python tail_fix.py noise  -> 產生底噪 tail_work/noise_bed.wav
"""
import json, sys
from pathlib import Path
import numpy as np, librosa, soundfile as sf, parselmouth
from parselmouth.praat import call
sys.path.insert(0, str(Path(__file__).parent))
import concat

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "routeB" / "tail_work"
SR = 24000
HOP = int(SR * 0.01)


def edb(y):
    rms = librosa.feature.rms(y=y, frame_length=HOP * 3, hop_length=HOP)[0]
    return 20 * np.log10(np.maximum(rms, 1e-8))


def trim_keep(y, tail=0.2, head=0.04):
    db = edb(y)
    thr = max(np.percentile(db, 95) - 35, np.percentile(db, 5) + 6)
    idx = np.where(db > thr)[0]
    a = max(0, idx[0] * HOP - int(head * SR))
    b = min(len(y), (idx[-1] + 1) * HOP + int(tail * SR))
    return y[a:b], (idx[-1] + 1) * HOP - a  # 語音結束位置


def stretch_tail(y, target=0.31, flat=0.0):
    """延長最後一個音；flat>0 時把句尾音高下降量縮掉 flat 比例。回傳新 y。"""
    y, end = trim_keep(y)
    db = edb(y)
    n_end = end // HOP
    lo = max(0, n_end - 40)
    peak = lo + int(np.argmax(db[lo:n_end]))
    start = max(0, peak * HOP - int(0.08 * SR))
    head, reg = y[:start], y[start:]
    reg_len = len(reg) / SR
    # 區段長度（含自然尾巴）以語音部分計
    speech_len = (end - start) / SR
    p2e = (end - peak * HOP) / SR + 0.02
    fac = float(np.clip(target / max(p2e, 0.05), 1.0, 2.0))
    snd = parselmouth.Sound(reg.astype(np.float64), SR)
    man = call(snd, "To Manipulation", 0.01, 75, 400)
    if flat > 0:
        pt = call(man, "Extract pitch tier")
        n = call(pt, "Get number of points")
        if n >= 4:
            ts = [call(pt, "Get time from index", i + 1) for i in range(n)]
            fs = [call(pt, "Get value at index", i + 1) for i in range(n)]
            # 只動最後約 250 ms 內的點
            T = speech_len
            wf = [f for t, f in zip(ts, fs) if t > T - 0.3]
            fall = 12 * np.log2(np.mean(wf[-2:]) / np.mean(wf[:2])) if len(wf) >= 4 else 0
            print("fall", round(fall,2), len(wf)) if __import__("os").environ.get("DBG") else None
            fall = max(fall, -6.0)
            if fall < 0:
                t0, t1 = T - 0.3, min(ts[-1], T)
                npt = call("Create PitchTier", "p", 0, snd.xmax)
                for t, f in zip(ts, fs):
                    w = float(np.clip((t - t0) / max(t1 - t0, 1e-3), 0, 1))
                    call(npt, "Add point", t, f * 2 ** ((-fall * flat * w) / 12))
                call([man, npt], "Replace pitch tier")
    dt = call("Create DurationTier", "d", 0, snd.xmax)
    call(dt, "Add point", 0.0, fac)
    # 只拉伸「峰前 80ms 之後」到語音結束的部分；其後的尾巴照常
    s_end = speech_len
    call(dt, "Add point", 0.001, fac)
    call(dt, "Add point", snd.xmax, fac)
    call([man, dt], "Replace duration tier")
    out = call(man, "Get resynthesis (overlap-add)")
    reg2 = out.values[0].astype(np.float32)
    xf = int(0.01 * SR)
    if len(head) > xf:
        w = np.linspace(0, 1, xf)
        mix = head[-xf:] * (1 - w) + reg2[:xf] * w
        return np.concatenate([head[:-xf], mix, reg2[xf:]]).astype(np.float32)
    return reg2


def cos_fade(y, sr, ms=10):
    y = y.copy()
    n = int(sr * ms / 1000)
    y[:n] *= np.linspace(0, 1, n)
    m = min(int(sr * 0.25), len(y) // 2)
    y[-m:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, m)))
    return y


def prep():
    picks = json.load(open(ROOT / "routeB/eval/candidates_ov_slow.json", encoding="utf-8"))["methods"]["ov_design_slow_design"]["pick"]
    files = [ROOT / p["file"] for p in picks]
    concat.trim = lambda y, sr, margin=0.04: y
    concat.fade = cos_fade
    for name, flat in (("T1", 0.0), ("T2", 0.75)):
        d = W / f"{name}_sent"; d.mkdir(parents=True, exist_ok=True)
        outs = []
        for i, f in enumerate(files, 1):
            y, _ = librosa.load(str(f), sr=SR)
            y2 = stretch_tail(y, flat=flat)
            p = d / f"s{i}.wav"; sf.write(p, y2, SR); outs.append(p)
        y = concat.build(outs)
        sf.write(W / f"{name}_src.wav", y, SR, subtype="PCM_16")
        print(name, len(y) / SR, concat.LAST_INFO)


if __name__ == "__main__":
    pass


def make_noise():
    """從阿公原始錄音挑最安靜的幾段（去掉低頻），交叉淡化串成可循環的底噪。"""
    import scipy.signal as ss
    y, sr = librosa.load(str(ROOT / "stage1/wav/male_voice_only.wav"), sr=SR)
    sos = ss.butter(4, 120, "hp", fs=SR, output="sos")
    y = ss.sosfilt(sos, y).astype(np.float32)
    win = int(0.6 * SR)
    hop = int(0.1 * SR)
    r = np.array([np.sqrt(np.mean(y[i:i + win] ** 2)) for i in range(0, len(y) - win, hop)])
    order = np.argsort(r)
    chosen = []
    for k in order:
        if all(abs(k - c) * 0.1 > 1.0 for c in chosen):
            chosen.append(k)
        if len(chosen) == 4:
            break
    print("quiet rms dBFS", [round(20 * np.log10(r[k] + 1e-9), 1) for k in chosen])
    xf = int(0.2 * SR)
    out = np.zeros(0, np.float32)
    for k in chosen:
        seg = y[k * hop:k * hop + win].copy()
        seg[:xf] *= np.linspace(0, 1, xf); seg[-xf:] *= np.linspace(1, 0, xf)
        if len(out) == 0:
            out = seg
        else:
            out = np.concatenate([out[:-xf], out[-xf:] + seg[:xf], seg[xf:]])
    # 頭尾相接成環
    wrap = out[-xf:] + out[:xf]
    out = np.concatenate([wrap, out[xf:-xf]])
    out = out / (np.sqrt(np.mean(out ** 2)) + 1e-9)  # RMS=1
    sf.write(W / "noise_bed.wav", out * 0.01, SR)
    return out


def add_bed(y, sr, dbfs=-60.0, seed=0):
    import soundfile
    nb, _ = librosa.load(str(W / "noise_bed.wav"), sr=sr)
    nb = nb / (np.sqrt(np.mean(nb ** 2)) + 1e-9) * 10 ** (dbfs / 20)
    n = int(np.ceil(len(y) / len(nb))) + 1
    big = np.tile(nb, n)
    off = np.random.RandomState(seed).randint(0, len(nb))
    bed = big[off:off + len(y)]
    f = int(0.5 * sr)
    bed[:f] *= np.linspace(0, 1, f); bed[-f:] *= np.linspace(1, 0, f)
    return (y + bed).astype(np.float32)




def post_t3():
    y, sr = sf.read(W / "vc_T1.wav", dtype="float32")
    sf.write(W / "vc_T3.wav", add_bed(y, sr), sr, subtype="PCM_16")


def t4_prep():
    """每句：T1 處理後的句子前後墊 0.3 s 底噪（-50 dBFS）。"""
    d = W / "T4_in"; d.mkdir(exist_ok=True)
    for i in range(1, 9):
        y, _ = librosa.load(str(W / "T1_sent" / f"s{i}.wav"), sr=SR)
        pad = np.zeros(int(0.3 * SR), np.float32)
        z = np.concatenate([pad, y, pad])
        z = add_bed(z, SR, dbfs=-50.0, seed=i)
        # 句子本身不加（避免雙重加底噪）：只在墊的部分有底噪，用相加後句內也有 -50 dB，影響可忽略
        sf.write(d / f"s{i}.wav", z, SR)


def t4_post():
    d = W / "T4_out"
    outs = []
    concat.trim = lambda y, sr, margin=0.04: y
    concat.fade = cos_fade
    od = W / "T4_cut"; od.mkdir(exist_ok=True)
    for i in range(1, 9):
        y, sr = sf.read(d / f"vc_s{i}.wav", dtype="float32")
        n = int(0.3 * sr)
        y = y[n:-n]
        # 去掉殘留底噪造成的尾端：維持原本長度
        sf.write(od / f"s{i}.wav", y, sr); outs.append(od / f"s{i}.wav")
    y = concat.build(outs)
    sf.write(W / "vc_T4.wav", y, SR, subtype="PCM_16")
    print(len(y) / SR, concat.LAST_INFO)


if __name__ == "__main__":
    {"prep": prep, "noise": make_noise, "t3": post_t3, "t4prep": t4_prep, "t4post": t4_post}[sys.argv[1]]()
