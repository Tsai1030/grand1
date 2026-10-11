"""把同一段來源用多個模型／存檔點轉換，再跑評估，彙整成一張表。

用 envs/seedvc 的 python 執行。轉換每次開新的子行程（run_infer.py，固定種子 1234）；
評估呼叫 eval/ 的工具（envs/prep 的 python，--device cpu）以及 routeA/scripts/f0_corr.py。

用法：
  # 掃描訓練資料夾的所有存檔點
  python convert_and_eval.py --runs routeA/runs/v1_denoise routeA/runs/v2_denoise --out-dir routeA/finetune
  # 指定 jobs（JSON 清單）
  python convert_and_eval.py --jobs routeA/best_checkpoints.json --out-dir routeA/final/xxx
  # 只重算評估（不轉換）
  python convert_and_eval.py --eval-only --out-dir routeA/finetune

jobs 每一項的欄位：
  label（輸出檔名）、model（v1 / v2）、ckpt（存檔點；不寫＝官方預設零樣本模型）、config（v1 存檔點旁的 yml）、
  style（v2：true＝convert-style 開）、ref（參考音，可用專案根目錄的相對路徑）、source（不寫＝--source）、
  pause_cap（秒；轉換前先把來源中間的長停頓縮到這個長度）、atempo（轉換前整段變速倍數，保持音高）、
  length_adjust（模型內建的長度調整）、seed（預設 1234）、note（說明）；
  v1 另可覆寫 diffusion_steps、cfg（inference-cfg-rate，預設 0.7）、fp16（預設 true）、f0_condition、auto_f0_adjust。
共通選項：--source（預設 inputs/taiwanese_source_joined.wav，找不到就用根目錄那份）、--ref（預設 inputs/voice_reference_15s.wav；
          加 --ref-override 時蓋過 jobs 裡的 ref）、--steps（擴散步數，預設 30）、--skip-existing、--no-eval。
輸出：<out-dir>/*.wav、<out-dir>/eval/*.json、<out-dir>/summary.json、<out-dir>/summary.tsv
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY_SEEDVC = os.path.join(ROOT, "envs", "seedvc", "Scripts", "python.exe")
PY_PREP = os.path.join(ROOT, "envs", "prep", "Scripts", "python.exe")
RUN_INFER = os.path.join(HERE, "run_infer.py")
F0CORR = os.path.join(HERE, "f0_corr.py")


def default_source():
    for p in [os.path.join(ROOT, "inputs", "taiwanese_source_joined.wav"),
              os.path.join(ROOT, "taiwanese_source_joined.wav")]:
        if os.path.exists(p):
            return p
    raise SystemExit("找不到 taiwanese_source_joined.wav（inputs/ 或根目錄）")


def env():
    e = dict(os.environ)
    ff = open(os.path.join(ROOT, "tools", "FFMPEG_PATH.txt"), encoding="utf-8-sig").read().strip().splitlines()[0]
    e["PATH"] = os.path.dirname(ff) + os.pathsep + e.get("PATH", "")
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def jobs_from_run(run_dir):
    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)
    jobs = []
    stats = {}
    sp = os.path.join(run_dir, "train_stats.json")
    if os.path.exists(sp):
        stats = json.load(open(sp, encoding="utf-8"))
    v1_cfg = glob.glob(os.path.join(run_dir, "config_dit_mel_*.yml"))
    if v1_cfg:  # v1
        cfg = v1_cfg[0]
        for p in sorted(glob.glob(os.path.join(run_dir, "DiT_epoch_*_step_*.pth"))):
            step = int(re.search(r"step_(\d+)", p).group(1))
            jobs.append({"label": f"{name}_step{step:04d}", "model": "v1", "ckpt": p, "config": cfg, "step": step, "run": name})
        ft = os.path.join(run_dir, "ft_model.pth")
        if os.path.exists(ft):
            a = stats.get("args", [])
            step = int(a[a.index("--max-steps") + 1]) if "--max-steps" in a else -1
            jobs.append({"label": f"{name}_step{step:04d}", "model": "v1", "ckpt": ft, "config": cfg, "step": step, "run": name})
    else:  # v2（只微調 CFM；AR 用預設）
        for p in sorted(glob.glob(os.path.join(run_dir, "CFM_epoch_*_step_*.pth"))):
            step = int(re.search(r"step_(\d+)", p).group(1))
            for style in (False, True):
                jobs.append({"label": f"{name}_step{step:04d}_style-{'on' if style else 'off'}", "model": "v2",
                             "ckpt": p, "style": style, "step": step, "run": name})
    jobs.sort(key=lambda j: (j["run"], j["step"], j.get("style", False)))
    return jobs


def infer_args(job, steps):
    steps = job.get("diffusion_steps", steps)
    if job["model"] == "v1":
        # 可在 job 裡覆寫：cfg（inference-cfg-rate）、fp16、f0_condition（True＝44.1kHz 歌聲模型）、auto_f0_adjust
        a = ["--diffusion-steps", str(steps), "--inference-cfg-rate", str(job.get("cfg", 0.7)),
             "--f0-condition", str(bool(job.get("f0_condition", False))),
             "--auto-f0-adjust", str(bool(job.get("auto_f0_adjust", False))),
             "--fp16", str(bool(job.get("fp16", True)))]
        if job.get("ckpt"):
            a += ["--checkpoint", job["ckpt"], "--config", job["config"]]
    else:
        a = ["--diffusion-steps", str(steps), "--intelligibility-cfg-rate", "0.7", "--similarity-cfg-rate", "0.7",
             "--convert-style", "true" if job.get("style") else "false", "--top-p", "0.9", "--temperature", "1.0",
             "--repetition-penalty", "1.0"]
        if job.get("ckpt"):
            a += ["--cfm-checkpoint-path", job["ckpt"]]
    return a


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def convert(jobs, source, ref, out_dir, steps, skip_existing, ref_override=False):
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    for j in jobs:
        out = os.path.join(out_dir, j["label"] + ".wav")
        if skip_existing and os.path.exists(out):
            print(f"[skip] {j['label']}")
            continue
        # 每個 job 可以另外指定 source／ref（相對路徑以專案根目錄為準）與 length_adjust（只限 v1/v2 的原生長度調整）
        src_j = _abs(j["source"]) if j.get("source") else source
        ref_j = _abs(j["ref"]) if (j.get("ref") and not ref_override) else ref
        # 轉換前處理（不改音高）：pause_cap＝只縮短停頓；atempo＝整段變速
        if j.get("pause_cap") or j.get("atempo"):
            sys.path.insert(0, HERE)
            import make_variants as mv
            vdir = os.path.join(out_dir, "sources")
            os.makedirs(vdir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(src_j))[0]
            if j.get("pause_cap"):
                vpath = os.path.join(vdir, f"{stem}_pausecap{j['pause_cap']}.wav")
                if not os.path.exists(vpath):
                    mv.pause_cap(src_j, float(j["pause_cap"]), vpath)
            else:
                vpath = os.path.join(vdir, f"{stem}_atempo{j['atempo']}.wav")
                if not os.path.exists(vpath):
                    mv.atempo(src_j, float(j["atempo"]), vpath)
            src_j = vpath
        if j.get("ckpt"):
            j["ckpt"] = _abs(j["ckpt"])
        if j.get("config"):
            j["config"] = _abs(j["config"])
        extra = ["--length-adjust", str(j["length_adjust"])] if j.get("length_adjust") else []
        cmd = [PY_SEEDVC, RUN_INFER, "--model", j["model"], "--out", out, "--seed", str(j.get("seed", 1234)), "--",
               "--source", src_j, "--target", ref_j] + infer_args(j, steps) + extra
        t = time.time()
        with open(os.path.join(out_dir, "logs", j["label"] + ".log"), "w", encoding="utf-8") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env())
        print(f"[{'ok' if r.returncode == 0 else 'FAIL'} {time.time()-t:5.1f}s] {j['label']}", flush=True)
        if os.path.exists(out + ".json"):
            info = json.load(open(out + ".json", encoding="utf-8"))
            info.update({k: j.get(k) for k in ("label", "model", "ckpt", "config", "style", "step", "run",
                                                "length_adjust", "pause_cap", "atempo", "note")})
            json.dump(info, open(out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _meta(w):
    return json.load(open(w + ".json", encoding="utf-8")) if os.path.exists(w + ".json") else {}


def _arg(meta, flag):
    a = meta.get("args") or []
    return a[a.index(flag) + 1] if flag in a else None


def evaluate(out_dir, source):
    ev = os.path.join(out_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(out_dir, "*.wav")))
    if not wavs:
        raise SystemExit("沒有可評估的 wav")
    e = env()
    subprocess.run([PY_PREP, os.path.join(ROOT, "eval", "speaker_sim.py"), "--agong-halves", "--agong-denoise-halves",
                    "--device", "cpu", "--test"] + wavs + ["--json", os.path.join(ev, "speaker_sim.json")],
                   check=True, env=e, cwd=ROOT)
    subprocess.run([PY_PREP, os.path.join(ROOT, "eval", "prosody_check.py")] + wavs +
                   ["--baseline", os.path.join(ROOT, "eval", "baseline.json"), "--json", os.path.join(ev, "prosody.json")],
                   check=True, env=e, cwd=ROOT)
    # F0 相關：每個輸出跟「它實際用的來源」比（變速／縮停頓的版本跟處理後的來源比）
    groups = {}
    for w in wavs:
        groups.setdefault(_arg(_meta(w), "--source") or source, []).append(w)
    rows = []
    for i, (src, ws) in enumerate(sorted(groups.items())):
        jp = os.path.join(ev, f"f0corr_{i}.json")
        subprocess.run([PY_SEEDVC, F0CORR, "--ref", src, "--hyp"] + ws + ["--json", jp], check=True, env=e, cwd=ROOT)
        rows += json.load(open(jp, encoding="utf-8"))["rows"]
    json.dump({"rows": rows, "sources": sorted(groups)}, open(os.path.join(ev, "f0corr.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    subprocess.run([PY_SEEDVC, os.path.join(HERE, "rhythm.py")] + wavs + ["--json", os.path.join(ev, "rhythm.json")],
                   check=True, env=e, cwd=ROOT)
    subprocess.run([PY_SEEDVC, os.path.join(HERE, "hf_quality.py")] + wavs + ["--json", os.path.join(ev, "hf_quality.json")],
                   check=True, env=e, cwd=ROOT)
    summarize(out_dir, wavs)


def summarize(out_dir, wavs):
    ev = os.path.join(out_dir, "eval")
    sim = json.load(open(os.path.join(ev, "speaker_sim.json"), encoding="utf-8"))
    pro = json.load(open(os.path.join(ev, "prosody.json"), encoding="utf-8"))
    f0 = json.load(open(os.path.join(ev, "f0corr.json"), encoding="utf-8"))
    rh = json.load(open(os.path.join(ev, "rhythm.json"), encoding="utf-8"))
    rh_by = {r["file"]: r for r in rh["files"]}
    hp = os.path.join(ev, "hf_quality.json")
    hf_by = {r["file"]: r for r in json.load(open(hp, encoding="utf-8"))["rows"]} if os.path.exists(hp) else {}
    f0rows = {r["hyp"]: r for r in f0["rows"]}
    pro_by_name = {os.path.basename(k): v for k, v in pro["files"].items()}
    rows = []
    for w in wavs:
        n = os.path.basename(w)
        s = sim["scores"][n]
        raw = [v for k, v in s.items() if "原始" in k]
        den = [v for k, v in s.items() if "去噪" in k]
        p = pro_by_name[n]
        f = f0rows[n]
        meta = _meta(w)
        r_ = rh_by.get(n, {})
        rows.append({
            "file": n,
            "run": meta.get("run"), "step": meta.get("step"), "model": meta.get("model"), "style": meta.get("style"),
            "source": os.path.basename(_arg(meta, "--source") or ""), "ref": os.path.basename(_arg(meta, "--target") or ""),
            "length_adjust": _arg(meta, "--length-adjust"),
            "speech_rate": r_.get("speech_rate"), "articulation_rate": r_.get("articulation_rate"),
            "pause_ratio": r_.get("pause_ratio"), "mean_pause_sec": r_.get("mean_pause_sec"),
            "hf_ratio_db": hf_by.get(n, {}).get("hf_ratio_db"), "hf_flatness": hf_by.get(n, {}).get("hf_flatness"),
            "hf_peakiness_db": hf_by.get(n, {}).get("hf_peakiness_db"),
            "sim_raw_first": round(s["阿公原始[前半]"], 4), "sim_raw_second": round(s["阿公原始[後半]"], 4),
            "sim_raw_mean": round(sum(raw) / len(raw), 4),
            "sim_dn_first": round(s["阿公去噪[前半]"], 4), "sim_dn_second": round(s["阿公去噪[後半]"], 4),
            "sim_dn_mean": round(sum(den) / len(den), 4),
            "pitch_jump_ratio": round(p["pitch_jump_ratio"], 4), "loud_per_10s": round(p["loud_per_10s"], 2),
            "f0_median_hz": round(p["f0_median_hz"], 1),
            "f0_r_global": f["r_global"], "f0_r_local": f["r_local"], "f0_rmse_st": f["rmse_st"],
            "vuv_agree": f["vuv_agree"], "sec": p["duration_sec"],
            "infer_wall_sec": meta.get("wall_sec_including_model_load"),
            "infer_peak_reserved_GB": meta.get("torch_max_reserved_GB"),
        })
    json.dump({"refs": list(sim["scores"][os.path.basename(wavs[0])].keys()), "rows": rows},
              open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cols = list(rows[0].keys())
    with open(os.path.join(out_dir, "summary.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    print("\t".join(["file", "raw1", "raw2", "dn1", "dn2", "dnMean", "jump", "loud10", "r_glob", "r_loc", "syl/s", "pause", "hfdB", "hfflat"]))
    for r in rows:
        print("\t".join(str(x) for x in [r["file"], r["sim_raw_first"], r["sim_raw_second"], r["sim_dn_first"],
                                          r["sim_dn_second"], r["sim_dn_mean"], r["pitch_jump_ratio"], r["loud_per_10s"],
                                          r["f0_r_global"], r["f0_r_local"], r["speech_rate"], r["pause_ratio"],
                                          r["hf_ratio_db"], r["hf_flatness"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--jobs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source")
    ap.add_argument("--ref", default=os.path.join(ROOT, "inputs", "voice_reference_15s.wav"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--ref-override", action="store_true")
    args = ap.parse_args()
    source = os.path.abspath(args.source) if args.source else default_source()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not args.eval_only:
        jobs = []
        for r in args.runs:
            jobs += jobs_from_run(r)
        if args.jobs:
            jobs += json.load(open(args.jobs, encoding="utf-8-sig"))
        print(f"{len(jobs)} 個轉換；來源 {source}；參考 {args.ref}")
        convert(jobs, source, os.path.abspath(args.ref), out_dir, args.steps, args.skip_existing, args.ref_override)
    if not args.no_eval:
        evaluate(out_dir, source)


if __name__ == "__main__":
    main()
