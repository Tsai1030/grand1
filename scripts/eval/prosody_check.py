"""順不順的檢查：音高突然跳動的比例、特別大聲的字。

例：
  envs\\prep\\Scripts\\python.exe eval\\prosody_check.py routeA\\out\\ --json routeA\\prosody.json --baseline eval\\baseline.json

指標（預設參數；改參數時阿公參考值也要用同樣參數重算）：
  pitch_jump_ratio   相鄰兩個有聲幀（hop 10 ms）之間音高變化超過 --jump-st 半音（預設 3）的比例。
                     有聲幀＝librosa.pyin 判為有聲（fmin 60 Hz、fmax 500 Hz、16 kHz、frame 1024、hop 160）；
                     幀間變化用逐幀獨立的 librosa.yin（同參數）計算（pyin 的平滑會把跳動抹掉）。
                     自然語音幀間變化多在 1 半音內，超過 3 半音的幾乎都是破音／八度跳，結果對 3–5 半音不敏感。
  loud_words         短時響度（100 ms 視窗 RMS，hop 10 ms）高於「有聲幀響度中位數 + --loud-db（預設 8 dB）」
                     的峰數（峰之間至少相隔 150 ms、突出度至少 3 dB）。
  loud_per_10s       每 10 秒「有聲時間」的大聲字數（不同長度的音檔才能比）。
  多個檔案的「合計」：跳動比例＝全部跳動幀數／全部有聲幀對數；大聲字＝全部峰數／全部有聲秒數 ×10。
判讀：不高於阿公本人錄音的水準（eval/baseline.json 的 prosody.agong_reference）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_io import expand_paths, load_audio  # noqa: E402

SR = 16000
HOP = 160          # 10 ms
FRAME = 1024       # 64 ms（pyin）
RMS_WIN = 1600     # 100 ms
PARAMS = {"sr": SR, "hop": HOP, "f0_frame": FRAME, "rms_win": RMS_WIN, "fmin": 60.0, "fmax": 500.0,
          "voicing": "librosa.pyin", "jump_f0": "librosa.yin", "peak_distance_frames": 15,
          "peak_prominence_db": 3.0}


def analyze(y: np.ndarray, jump_st: float = 3.0, loud_db: float = 8.0,
            fmin: float = 60.0, fmax: float = 500.0) -> dict:
    import librosa
    from scipy.signal import find_peaks

    dur = len(y) / SR
    # 有聲判斷與 F0 中位數用 pyin（HMM 平滑）；跳動用逐幀獨立的 YIN 估計，
    # 因為 pyin 的 Viterbi 會限制每幀變化量（約 2 半音），把破音／八度跳動抹平、量不到。
    f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=FRAME, hop_length=HOP)
    f0_raw = librosa.yin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=FRAME, hop_length=HOP)
    n0 = min(len(f0), len(f0_raw))
    f0, vflag, f0_raw = f0[:n0], vflag[:n0], f0_raw[:n0]
    voiced = vflag & np.isfinite(f0)
    pairs = voiced[:-1] & voiced[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        d_st = 12.0 * np.abs(np.log2(f0_raw[1:] / f0_raw[:-1]))
    d_st = d_st[pairs]
    n_pairs = int(pairs.sum())
    n_jumps = int((d_st > jump_st).sum())
    f0v = f0[voiced]

    # 短時響度（RMS dB），center=True 讓幀與 pyin 對齊
    rms = librosa.feature.rms(y=y, frame_length=RMS_WIN, hop_length=HOP, center=True)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    n = min(len(db), len(voiced))
    db, v2 = db[:n], voiced[:n]
    voiced_sec = float(v2.sum() * HOP / SR)
    if v2.sum() >= 5:
        med = float(np.median(db[v2]))
        peaks, _ = find_peaks(db, height=med + loud_db, distance=15, prominence=3.0)
        loud = int(len(peaks))
        loud_times = [round(float(p * HOP / SR), 2) for p in peaks]
        max_over = float(db[v2].max() - med)
    else:
        med, loud, loud_times, max_over = float("nan"), 0, [], float("nan")

    return {
        "duration_sec": round(dur, 3),
        "voiced_sec": round(voiced_sec, 3),
        "voiced_pairs": n_pairs,
        "pitch_jumps": n_jumps,
        "pitch_jump_ratio": (n_jumps / n_pairs) if n_pairs else float("nan"),
        "f0_median_hz": float(np.median(f0v)) if len(f0v) else float("nan"),
        "f0_p5_p95_hz": [float(np.percentile(f0v, 5)), float(np.percentile(f0v, 95))] if len(f0v) else None,
        "voiced_median_db": med,
        "loud_words": loud,
        "loud_per_10s": (loud / voiced_sec * 10) if voiced_sec > 0 else float("nan"),
        "loudest_over_median_db": max_over,
        "loud_word_times_sec": loud_times,
    }


def pooled(results: dict) -> dict:
    r = list(results.values())
    pairs = sum(x["voiced_pairs"] for x in r)
    jumps = sum(x["pitch_jumps"] for x in r)
    vsec = sum(x["voiced_sec"] for x in r)
    loud = sum(x["loud_words"] for x in r)
    ratios = [x["pitch_jump_ratio"] for x in r if np.isfinite(x["pitch_jump_ratio"])]
    lp10 = [x["loud_per_10s"] for x in r if np.isfinite(x["loud_per_10s"])]
    return {
        "files": len(r),
        "duration_sec": round(sum(x["duration_sec"] for x in r), 3),
        "voiced_sec": round(vsec, 3),
        "pitch_jump_ratio": jumps / pairs if pairs else float("nan"),
        "loud_words": loud,
        "loud_per_10s": loud / vsec * 10 if vsec else float("nan"),
        "per_file_pitch_jump_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        "per_file_pitch_jump_ratio_p90": float(np.percentile(ratios, 90)) if ratios else float("nan"),
        "per_file_loud_per_10s_median": float(np.median(lp10)) if lp10 else float("nan"),
        "per_file_loud_per_10s_p90": float(np.percentile(lp10, 90)) if lp10 else float("nan"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="音高跳動比例、特別大聲的字")
    ap.add_argument("inputs", nargs="+", help="音檔或資料夾（可多個）")
    ap.add_argument("--jump-st", type=float, default=3.0, help="相鄰有聲幀跳超過幾個半音算「突然跳動」")
    ap.add_argument("--loud-db", type=float, default=8.0, help="比有聲中位數響幾 dB 以上的峰算「特別大聲」")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--json", type=Path, help="結果另存 JSON")
    ap.add_argument("--baseline", type=Path, help="eval/baseline.json：一併列出阿公參考值並比較")
    args = ap.parse_args(argv)

    files = expand_paths(args.inputs, args.recursive)
    results = {}
    for f in files:
        y, _ = load_audio(f, SR)
        results[str(f)] = analyze(y, args.jump_st, args.loud_db)
    agg = pooled(results)

    ref = None
    if args.baseline and args.baseline.exists():
        b = json.loads(args.baseline.read_text(encoding="utf-8"))
        ref = b.get("prosody", {}).get("agong_reference")
        if ref and (ref["params"]["jump_st"] != args.jump_st or ref["params"]["loud_db"] != args.loud_db):
            print("# 警告：參數跟 baseline 不同，比較無意義", file=sys.stderr)

    print(f"# 參數：jump_st={args.jump_st} 半音、loud_db={args.loud_db} dB")
    print("檔案\t秒數\t有聲秒\t音高跳動比例\t大聲字\t每10秒大聲字\tF0中位(Hz)")
    for f, r in results.items():
        print(f"{Path(f).name}\t{r['duration_sec']:.2f}\t{r['voiced_sec']:.2f}\t{r['pitch_jump_ratio']:.4f}\t"
              f"{r['loud_words']}\t{r['loud_per_10s']:.2f}\t{r['f0_median_hz']:.1f}")
    print(f"合計({agg['files']}檔)\t{agg['duration_sec']:.2f}\t{agg['voiced_sec']:.2f}\t"
          f"{agg['pitch_jump_ratio']:.4f}\t{agg['loud_words']}\t{agg['loud_per_10s']:.2f}\t")
    verdict = None
    if ref:
        rp = ref["pooled"]
        print(f"阿公參考\t{rp['duration_sec']:.2f}\t{rp['voiced_sec']:.2f}\t{rp['pitch_jump_ratio']:.4f}\t"
              f"{rp['loud_words']}\t{rp['loud_per_10s']:.2f}\t")
        verdict = {
            "pitch_jump_ratio_ok": bool(agg["pitch_jump_ratio"] <= rp["pitch_jump_ratio"]),
            "loud_per_10s_ok": bool(agg["loud_per_10s"] <= rp["loud_per_10s"]),
        }
        print(f"# 音高跳動 {'≤' if verdict['pitch_jump_ratio_ok'] else '>'} 阿公；"
              f"大聲字 {'≤' if verdict['loud_per_10s_ok'] else '>'} 阿公")

    if args.json:
        out = {"params": PARAMS | {"jump_st": args.jump_st, "loud_db": args.loud_db},
               "files": results, "pooled": agg, "vs_agong": verdict}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"# JSON 已存：{args.json}")
    return results, agg


if __name__ == "__main__":
    main()
