"""逐句候選評估 → 每句挑 seed → 串成整段 → 整段評估。用 envs/prep 的 python。

方法資料夾：每個裡面有 s<句號>_seed<k>.wav。名稱自動取（cosyvoice/<指令>/<prompt> → cv_<指令>_<prompt>）。
逐句指標：
  sim_dn   ＝對「阿公去噪版」前半、後半的 CAM++ 相似度平均（eval/speaker_sim.py 的同一套模型與切點；同一人基準 0.901）
  sim_raw  ＝對「阿公原始錄音」前半、後半的平均（同一人基準 0.939；乾淨音上限約 0.8）
  f0_std_st、f0_range_st ＝語調起伏（routeB/scripts/f0_stats.py）
  jump     ＝音高突然跳動比例（eval/prosody_check.py 的 analyze，> 3 半音）
  syl_rate ＝音節數 ÷ 去頭尾靜音後的長度
  ASR      ＝asr_check.py 的結果（key_hit、problem）
挑 seed（每個方法、每句）：先排除 ASR「有問題」的（全部都有問題就全部留著並標記），
  再用分數＝0.4×min(f0_std/阿公基準, 1.2) ＋ 0.15×句尾走向接近度 ＋ 0.25×sim_dn/0.80 ＋ 0.2×key_hit − 2×max(jump−0.03, 0) 取最高。
  句尾走向接近度＝max(0, 1 − |final_st − 阿公中位數| / 4)。
  （語調起伏是主要條件；跳動比例太高＝破音或怪聲扣分）
整段：concat.py 串接 → routeB/passages/<方法>.wav，再量相似度、節奏（routeA/scripts/rhythm.py）、F0（3 秒片段中位數）、跳動。

用法：python eval_candidates.py --methods 資料夾 ... --asr asr1.json asr2.json --out routeB/eval/candidates.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(Path(__file__).parent))
from audio_io import halves  # noqa: E402
from speaker_sim import AGONG_DENOISE_WAV, AGONG_RAW_WAV, SpeakerEmbedder, cosine  # noqa: E402
from prosody_check import analyze as prosody_analyze  # noqa: E402
import f0_stats  # noqa: E402
import concat  # noqa: E402


def load_rhythm():
    spec = importlib.util.spec_from_file_location("rhythm", ROOT / "routeA" / "scripts" / "rhythm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def method_name(d: Path):
    rel = d.resolve().relative_to(ROOT / "routeB")
    parts = rel.parts
    pre = {"cosyvoice": "cv", "omnivoice": "ov", "seedvc": "vc"}.get(parts[0], parts[0])
    return "_".join([pre] + list(parts[1:]))


def f0_chunk_median(y16, chunk=3.0):
    vals = []
    for part in f0_stats.split_chunks(y16, chunk):
        s = f0_stats.stats(f0_stats.f0_of(part))
        if s:
            vals.append((s["f0_std_st"], s["f0_range_st"]))
    if not vals:
        return None, None
    a = np.array(vals)
    return round(float(np.median(a[:, 0])), 2), round(float(np.median(a[:, 1])), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--asr", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-concat", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    sents = {s["id"]: s for s in json.loads((ROOT / "routeB" / "sentences.json").read_text(encoding="utf-8"))}
    asr = {}
    for a in args.asr:
        if not Path(a).exists():
            print(f"略過不存在的 ASR 檔：{a}", flush=True)
            continue
        for r in json.loads(Path(a).read_text(encoding="utf-8"))["rows"]:
            asr[str((ROOT / r["file"]).resolve()).lower()] = r

    emb = SpeakerEmbedder("campplus", args.device)
    raw_h = halves(AGONG_RAW_WAV, "阿公原始")
    dn_h = halves(AGONG_DENOISE_WAV, "阿公去噪", split=raw_h[0].end)
    ref_raw = [emb.embed(h) for h in raw_h]
    ref_dn = [emb.embed(h) for h in dn_h]

    def sims(y16):
        e = emb.embed_array(y16)
        return (round(float(np.mean([cosine(e, r) for r in ref_dn])), 4),
                round(float(np.mean([cosine(e, r) for r in ref_raw])), 4))

    # 阿公基準（同一把尺）：12 段去噪片段切 3 秒的 F0 起伏中位數
    ag, ag_ct = [], []
    for f in sorted((ROOT / "stage1" / "segments").glob("*.wav")):
        y = librosa.load(str(f), sr=16000)[0]
        ag.append(f0_chunk_median(y))
        # 語調走向：在停頓處切成短句（約 1.5 秒以上），每句算句尾走向與斜率
        for part in f0_stats.split_chunks(y, 1.5):
            c = f0_stats.contour(part)
            if c:
                ag_ct.append(c)
    ag_std = float(np.median([a[0] for a in ag if a[0]]))
    ag_rng = float(np.median([a[1] for a in ag if a[1]]))
    ag_final = float(np.median([c["final_st"] for c in ag_ct]))
    ag_slope = float(np.median([c["slope_st_s"] for c in ag_ct]))
    print(f"阿公基準（3 秒片段）F0 std {ag_std:.2f} st、range {ag_rng:.2f} st；短句 {len(ag_ct)} 個：句尾 {ag_final:+.2f} st、斜率 {ag_slope:+.2f} st/s", flush=True)

    rhythm = load_rhythm()
    result = {"agong_f0_ref": {"f0_std_st": round(ag_std, 2), "f0_range_st": round(ag_rng, 2), "final_st": round(ag_final, 2),
                               "slope_st_s": round(ag_slope, 2), "n_phrases": len(ag_ct),
                               "final_st_all": [c["final_st"] for c in ag_ct]}, "methods": {}}
    for d in (Path(x).resolve() for x in args.methods):
        name = method_name(d)
        rows = []
        for f in sorted(d.glob("s*_seed*.wav")):
            sid = int(f.stem.split("_")[0][1:])
            seed = int(f.stem.split("seed")[1])
            y24, _ = librosa.load(str(f), sr=24000)
            yt = concat.trim(y24, 24000)
            y16 = librosa.resample(yt, orig_sr=24000, target_sr=16000)
            st = f0_stats.stats(f0_stats.f0_of(y16)) or {"f0_std_st": 0.0, "f0_range_st": 0.0, "f0_median_hz": 0.0}
            pr = prosody_analyze(y16)
            sdn, sraw = sims(y16)
            a = asr.get(str(f.resolve()).lower(), {})
            jump = pr["pitch_jump_ratio"] if np.isfinite(pr["pitch_jump_ratio"]) else 0.0
            ct = f0_stats.contour(y16) or {"final_st": 0.0, "slope_st_s": 0.0}
            rows.append({"file": str(f.relative_to(ROOT)), "id": sid, "seed": seed, "dur": round(len(yt) / 24000, 2),
                         "syl_rate": round(sents[sid]["syllables"] / max(len(yt) / 24000, 0.1), 2),
                         "f0_std_st": st["f0_std_st"], "f0_range_st": st["f0_range_st"], "f0_median_hz": st["f0_median_hz"],
                         "final_st": ct["final_st"], "slope_st_s": ct["slope_st_s"], "jump": round(jump, 4), "sim_dn": sdn, "sim_raw": sraw,
                         "asr": a.get("asr"), "key_hit": a.get("key_hit"), "asr_problem": a.get("problem")})
        # 挑 seed
        pick = []
        for sid in sorted({r["id"] for r in rows}):
            cand = [r for r in rows if r["id"] == sid]
            ok = [r for r in cand if r["asr_problem"] is False]
            pool, all_bad = (ok, False) if ok else (cand, True)
            for r in pool:
                kh = r["key_hit"] if r["key_hit"] is not None else 0.5
                like_final = max(0.0, 1 - abs(r["final_st"] - ag_final) / 4.0)   # 句尾走向越接近阿公越好
                r["score"] = round(0.4 * min(r["f0_std_st"] / ag_std, 1.2) + 0.15 * like_final + 0.25 * r["sim_dn"] / 0.80
                                   + 0.2 * kh - 2 * max(r["jump"] - 0.03, 0), 4)
            best = max(pool, key=lambda r: r["score"])
            best["picked"] = True
            pick.append({"id": sid, "file": best["file"], "seed": best["seed"], "all_seeds_asr_problem": all_bad})
        m = {"dir": str(d), "rows": rows, "pick": pick,
             "asr_problem_sentences_in_pick": sum(1 for p in pick if p["all_seeds_asr_problem"]),
             "asr_ok_seeds": f"{sum(1 for r in rows if r['asr_problem'] is False)}/{len(rows)}"}
        if not args.no_concat and len(pick) == len(sents):
            out = ROOT / "routeB" / "passages" / f"{name}.wav"
            y = concat.build([ROOT / p["file"] for p in pick], n_syl=sum(x["syllables"] for x in sents.values()))
            m["concat"] = dict(concat.LAST_INFO)
            out.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            sf.write(str(out), y, concat.SR, subtype="PCM_16")
            m["passage"] = passage_metrics(out, sims, rhythm)
            m["passage"]["text_rate"] = m["concat"]["text_rate"]
            pr_rows = [r for r in rows if r.get("picked")]
            m["passage"]["final_st_median"] = round(float(np.median([r["final_st"] for r in pr_rows])), 2)
        result["methods"][name] = m
        ps = m.get("passage", {})
        print(f"{name}\tASR OK seeds {m['asr_ok_seeds']}\t挑中有問題句 {m['asr_problem_sentences_in_pick']}\t"
              f"整段 {ps.get('sec')}s sim_dn {ps.get('sim_dn')} sim_raw {ps.get('sim_raw')} "
              f"F0std {ps.get('f0_std_st_chunk')} range {ps.get('f0_range_st_chunk')} final {ps.get('final_st_median')} "
              f"字/秒 {ps.get('text_rate')} rate {ps.get('speech_rate')} "
              f"pause {ps.get('pause_ratio')} jump {ps.get('jump')}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def passage_metrics(path, sims, rhythm):
    y16, _ = librosa.load(str(path), sr=16000)
    sdn, sraw = sims(y16)
    r = rhythm.analyze(str(path))
    s_std, s_rng = f0_chunk_median(y16)
    whole = f0_stats.stats(f0_stats.f0_of(y16))
    pr = prosody_analyze(y16)
    return {"file": str(Path(path).relative_to(ROOT)), "sec": round(len(y16) / 16000, 2), "sim_dn": sdn, "sim_raw": sraw,
            "speech_rate": r["speech_rate"], "articulation_rate": r["articulation_rate"], "pause_ratio": r["pause_ratio"],
            "mean_pause_sec": r["mean_pause_sec"], "f0_std_st_chunk": s_std, "f0_range_st_chunk": s_rng,
            "f0_std_st_whole": whole["f0_std_st"] if whole else None, "f0_median_hz": whole["f0_median_hz"] if whole else None,
            "jump": round(pr["pitch_jump_ratio"], 4), "loud_per_10s": round(pr["loud_per_10s"], 2)}


if __name__ == "__main__":
    main()
