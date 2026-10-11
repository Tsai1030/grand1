"""步驟 7–8：在停頓處切句（1–30 秒），主力版與原始版用同樣切點；聲紋品質檢查；響度統一。

用法：
  python stage1/scripts/segment_and_qc.py --main denoise
輸出：
  stage1/segments/agong_XX.wav        主力版片段（響度 -20 LUFS、真峰值 ≤ -1 dBFS、不削峰）
  stage1/segments_orig/agong_XX.wav   同切點的原始未增強版
  stage1/rejected/{segments,segments_orig}/   聲紋特別低的片段（移過去，不刪）
  stage1/segments.json                切點、秒數、聲紋、響度等細節
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from common import ORIG_WAV, ROOT, STAGE1, VAD_PARAMS, VERSIONS, vad_regions
from audio_io import load_audio
from speaker_sim import SpeakerEmbedder, cosine

# 切句參數
LONG_GAP = 0.8     # 停頓 ≥ 0.8 秒一定優先切
MIN_SEG = 3.0      # 片段（語音範圍）至少 3 秒，避免太碎
MAX_SEG = 12.0     # 超過 12 秒就在內部最長停頓再切（Seed-VC 上限 30 秒）
PAD_BEFORE = 0.25  # 語音前留 0.25 秒
PAD_AFTER = 0.30   # 語音後留 0.30 秒
# 響度
TARGET_LUFS = -20.0
PEAK_CEIL_DBFS = -1.0
# 品質檢查：相似度低於 median - K*MAD*1.4826 或低於絕對門檻者剔除
QC_K = 3.0
QC_ABS = 0.5


def choose_groups(regions):
    """把 VAD 語音區段分組成片段；回傳 [(i_start, i_end_inclusive), ...]。"""
    n = len(regions)
    gaps = [regions[i + 1][0] - regions[i][1] for i in range(n - 1)]  # gap i 在 region i 與 i+1 之間
    cuts: set[int] = set()

    def seg_len(a, b):
        return regions[b][1] - regions[a][0]

    def groups_from(c):
        bounds, start = [], 0
        for i in sorted(c):
            bounds.append((start, i))
            start = i + 1
        bounds.append((start, n - 1))
        return bounds

    def ok(c):
        return all(seg_len(a, b) >= MIN_SEG for a, b in groups_from(c))

    # 1) 長停頓優先（由長到短），兩邊都要 ≥ MIN_SEG
    for i in sorted(range(n - 1), key=lambda k: -gaps[k]):
        if gaps[i] >= LONG_GAP and ok(cuts | {i}):
            cuts.add(i)
    # 2) 過長的片段在內部挑停頓切：分數＝停頓長度 − 0.05×離中點秒數
    changed = True
    while changed:
        changed = False
        for a, b in groups_from(cuts):
            if seg_len(a, b) <= MAX_SEG:
                continue
            mid = (regions[a][0] + regions[b][1]) / 2
            cands = [i for i in range(a, b) if ok(cuts | {i})]
            if not cands:
                cands = [i for i in range(a, b)
                         if seg_len(a, i) >= 1.0 and seg_len(i + 1, b) >= 1.0]
            if not cands:
                continue
            best = max(cands, key=lambda i: gaps[i] - 0.05 * abs((regions[i][1] + regions[i + 1][0]) / 2 - mid))
            cuts.add(best)
            changed = True
            break
    return groups_from(cuts), gaps


def loudnorm(y, sr):
    meter = pyln.Meter(sr)  # ITU-R BS.1770-4
    lufs_in = meter.integrated_loudness(y)
    gain_db = TARGET_LUFS - lufs_in
    # 用 4 倍超取樣估真峰值，確保不削峰
    import soxr
    tp = np.max(np.abs(soxr.resample(y, sr, sr * 4))) + 1e-12
    peak_after = 20 * np.log10(tp) + gain_db
    limited = False
    if peak_after > PEAK_CEIL_DBFS:
        gain_db -= peak_after - PEAK_CEIL_DBFS
        limited = True
    out = (y * 10 ** (gain_db / 20)).astype(np.float32)
    return out, {"lufs_in": round(lufs_in, 2), "gain_db": round(gain_db, 2),
                 "lufs_out": round(meter.integrated_loudness(out), 2),
                 "true_peak_out_dbfs": round(20 * np.log10(tp) + gain_db, 2),
                 "peak_limited": limited}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="denoise", choices=sorted(VERSIONS), help="主力版本")
    ap.add_argument("--vad-src", default="denoise", choices=sorted(VERSIONS), help="用哪一版跑 VAD 找停頓")
    args = ap.parse_args()

    main_path, orig_path = VERSIONS[args.main], ORIG_WAV
    y_main, sr = load_audio(main_path)
    y_orig, sr_o = load_audio(orig_path)
    assert sr == sr_o, (sr, sr_o)
    n = min(len(y_main), len(y_orig))
    y_main, y_orig = y_main[:n], y_orig[:n]
    dur = n / sr

    regions = vad_regions(load_audio(VERSIONS[args.vad_src], 16000)[0])
    groups, gaps = choose_groups(regions)

    segs = []
    for k, (a, b) in enumerate(groups):
        s_speech, e_speech = regions[a][0], regions[b][1]
        prev_lim = 0.0 if a == 0 else (regions[a - 1][1] + regions[a][0]) / 2
        next_lim = dur if b == len(regions) - 1 else (regions[b][1] + regions[b + 1][0]) / 2
        s = max(prev_lim, s_speech - PAD_BEFORE, 0.0)
        e = min(next_lim, e_speech + PAD_AFTER, dur)
        segs.append({"id": f"agong_{k + 1:02d}", "start_sec": round(s, 3), "end_sec": round(e, 3),
                     "duration_sec": round(e - s, 3), "vad_regions": [a, b],
                     "max_internal_pause_sec": round(max(gaps[a:b]), 2) if b > a else 0.0})

    out_main, out_orig = STAGE1 / "segments", STAGE1 / "segments_orig"
    rej = STAGE1 / "rejected"
    for d in (out_main, out_orig, rej / "segments", rej / "segments_orig"):
        if d.exists():
            for f in d.glob("*.wav"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    for sg in segs:
        i0, i1 = int(round(sg["start_sec"] * sr)), int(round(sg["end_sec"] * sr))
        for tag, y, d in (("main", y_main, out_main), ("orig", y_orig, out_orig)):
            seg, info = loudnorm(y[i0:i1].copy(), sr)
            sf.write(str(d / f"{sg['id']}.wav"), seg, sr, subtype="PCM_16")
            sg[f"loudness_{tag}"] = info

    # 品質檢查：每段 vs 其餘片段平均聲紋（leave-one-out centroid）
    emb = SpeakerEmbedder("campplus", "auto")
    for tag, d in (("main", out_main), ("orig", out_orig)):
        E = []
        for sg in segs:
            y16, _ = load_audio(d / f"{sg['id']}.wav", 16000)
            e = emb.embed_array(y16)
            E.append(e / np.linalg.norm(e))
        E = np.stack(E)
        tot = E.sum(0)
        for i, sg in enumerate(segs):
            loo = (tot - E[i]) / (len(segs) - 1)
            sg[f"sim_to_others_{tag}"] = round(cosine(E[i], loo), 4)
            sg[f"sim_to_all_{tag}"] = round(cosine(E[i], tot / len(segs)), 4)

    sims = np.array([sg["sim_to_others_main"] for sg in segs])
    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med))) * 1.4826
    thr = max(QC_ABS, med - QC_K * mad) if mad > 0 else QC_ABS
    for sg in segs:
        reasons = []
        if sg["sim_to_others_main"] < thr:
            reasons.append(f"聲紋相似度 {sg['sim_to_others_main']:.3f} < 門檻 {thr:.3f}（可能混到別人或雜音太多）")
        if not (1.0 <= sg["duration_sec"] <= 30.0):
            reasons.append(f"長度 {sg['duration_sec']:.2f} 秒不在 1–30 秒")
        sg["rejected"] = bool(reasons)
        sg["reject_reasons"] = reasons
        if reasons:
            shutil.move(str(out_main / f"{sg['id']}.wav"), str(rej / "segments" / f"{sg['id']}.wav"))
            shutil.move(str(out_orig / f"{sg['id']}.wav"), str(rej / "segments_orig" / f"{sg['id']}.wav"))

    kept = [s for s in segs if not s["rejected"]]
    res = {
        "main_version": args.main, "main_path": str(main_path.relative_to(ROOT)),
        "orig_path": str(orig_path.relative_to(ROOT)), "sample_rate": sr, "source_duration_sec": round(dur, 3),
        "vad": {"model": "silero-vad", "on": args.vad_src, **VAD_PARAMS, "regions": len(regions)},
        "cut_params": {"long_gap": LONG_GAP, "min_seg": MIN_SEG, "max_seg": MAX_SEG,
                       "pad_before": PAD_BEFORE, "pad_after": PAD_AFTER},
        "loudness": {"target_lufs": TARGET_LUFS, "true_peak_ceiling_dbfs": PEAK_CEIL_DBFS,
                     "meter": "pyloudnorm BS.1770-4", "format": "PCM_16"},
        "qc": {"model": emb.conf["model_id"], "metric": "cosine to leave-one-out mean of other segments",
               "median": round(med, 4), "mad_scaled": round(mad, 4), "k": QC_K, "abs_floor": QC_ABS,
               "threshold": round(thr, 4)},
        "n_segments": len(segs), "n_kept": len(kept),
        "kept_total_sec": round(sum(s["duration_sec"] for s in kept), 3),
        "segments": segs,
    }
    (STAGE1 / "segments.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for sg in segs:
        print(f"{sg['id']} {sg['start_sec']:7.2f}-{sg['end_sec']:7.2f} {sg['duration_sec']:5.2f}s "
              f"sim={sg['sim_to_others_main']:.3f}/{sg['sim_to_others_orig']:.3f} "
              f"lufs={sg['loudness_main']['lufs_out']} tp={sg['loudness_main']['true_peak_out_dbfs']} "
              f"{'REJECT ' + ';'.join(sg['reject_reasons']) if sg['rejected'] else ''}")
    print(f"kept {len(kept)}/{len(segs)} total {res['kept_total_sec']}s thr={thr:.3f}")


if __name__ == "__main__":
    main()
