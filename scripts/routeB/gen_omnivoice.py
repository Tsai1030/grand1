"""k2-fsa/OmniVoice 逐句生成：輸入台羅（language=nan），阿公參考音複製音色。

OmniVoice 的「voice clone」需要參考音逐字稿（ref_text）。我們沒有阿公參考音的台語逐字稿，
所以預設 --ref-text-mode empty：ref_text 給空字串（不送逐字稿、也不讓它自動跑 Whisper），
參考音的 audio token 照樣當前綴。這樣模型不知道參考音講什麼，可能把句首吃掉，要靠 ASR 檢查。
因為空逐字稿時模型無法用「參考音語速」估長度，所以每句長度用 --rate（每秒音節數）自己給。

其他 ref-text 模式（對照用）：
  whisper：讓 OmniVoice 用 Whisper-large-v3-turbo 自動轉寫（台語會被寫成華語，內容不對）
  zh：用 stage1 的 Breeze 華語意思（內容大概對、語言不對）

用法：envs\\omnivoice\\Scripts\\python.exe routeB\\scripts\\gen_omnivoice.py --device cuda \
        --ref ref15=inputs\\voice_reference_15s.wav --seeds 1 2 3 --variant base
輸出：routeB/omnivoice/<variant>/<ref>/s<句號>_seed<k>.wav（24 kHz）＋ meta.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
ap.add_argument("--ref", action="append", required=True, help="名稱=音檔路徑")
ap.add_argument("--ref-text-mode", choices=["empty", "whisper", "zh", "custom"], default="empty")
ap.add_argument("--ref-text", default="", help="--ref-text-mode custom 時用")
ap.add_argument("--ids", type=int, nargs="*")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--variant", default="base")
ap.add_argument("--text-field", default="tailo", help="tailo（調符）或 tailo_num（數字調）")
ap.add_argument("--language", default="nan")
ap.add_argument("--rate", type=float, default=3.3, help="每秒音節數（含句內停頓）；<=0 表示讓模型自己估")
ap.add_argument("--num-step", type=int, default=32)
ap.add_argument("--guidance", type=float, default=2.0)
ap.add_argument("--no-preprocess-prompt", action="store_true")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--instruct", default=None, help="voice design 屬性（只在不給參考音時用），例如 'male, elderly, moderate pitch'")
args = ap.parse_args()

if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[2]
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

if args.device == "cpu":
    # torch 2.6 在 CUDA_VISIBLE_DEVICES=-1 時，torch.accelerator.current_accelerator() 會丟 RuntimeError，
    # transformers 5.x import 時會呼叫它 → 改成回傳 None（= 沒有加速器，用 CPU）
    torch.accelerator.current_accelerator = lambda *a, **k: None
from omnivoice import OmniVoice  # noqa: E402

if args.device == "cpu":
    torch.set_num_threads(args.threads)
assert (args.device == "cuda") == torch.cuda.is_available()

MODEL_DIR = ROOT / "routeB" / "models" / "OmniVoice"
sents = json.loads((ROOT / "routeB" / "sentences.json").read_text(encoding="utf-8"))
if args.ids:
    sents = [s for s in sents if s["id"] in args.ids]

t0 = time.time()
dtype = torch.float16 if args.device == "cuda" else torch.float32
model = OmniVoice.from_pretrained(str(MODEL_DIR), device_map=("cuda:0" if args.device == "cuda" else "cpu"), dtype=dtype)
load_sec = time.time() - t0
sr = model.sampling_rate
print(f"loaded {load_sec:.1f}s sr={sr}", flush=True)

ZH_REF = {}
try:
    asr = json.loads((ROOT / "stage1" / "asr.json").read_text(encoding="utf-8"))["text"]
    ZH_REF = asr
except Exception:
    pass

for rspec in args.ref:
    rname, rpath = rspec.split("=", 1)
    out_dir = ROOT / "routeB" / "omnivoice" / args.variant / rname
    out_dir.mkdir(parents=True, exist_ok=True)
    if rpath == "":   # 不給參考音：auto voice，或 --instruct 的 voice design（例如 "male, elderly, moderate pitch"）
        meta = {"model": "k2-fsa/OmniVoice", "weights_license": "CC-BY-NC（非商用）", "mode": "voice design" if args.instruct else "auto voice",
                "instruct": args.instruct, "language": args.language, "text_field": args.text_field, "rate": args.rate,
                "num_step": args.num_step, "guidance_scale": args.guidance, "device": args.device, "load_sec": round(load_sec, 1),
                "items": []}
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for s in sents:
            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                dur = (s["syllables"] / args.rate) if args.rate > 0 else None
                t2 = time.time()
                y = model.generate(text=s[args.text_field], language=args.language, duration=dur, instruct=args.instruct,
                                   num_step=args.num_step, guidance_scale=args.guidance)[0]
                fn = out_dir / f"s{s['id']}_seed{seed}.wav"
                sf.write(str(fn), y, sr, subtype="PCM_16")
                item = {"id": s["id"], "seed": seed, "text_in": s[args.text_field], "target_dur": round(dur, 2) if dur else None,
                        "file": fn.name, "dur_sec": round(len(y) / sr, 3), "gen_sec": round(time.time() - t2, 2)}
                meta["items"].append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
        if args.device == "cuda":
            meta["cuda_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            meta["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        continue
    rpath = str((ROOT / rpath).resolve()) if not os.path.isabs(rpath) else rpath
    if args.ref_text_mode == "empty":
        ref_text = ""
    elif args.ref_text_mode == "whisper":
        ref_text = None
    elif args.ref_text_mode == "zh":
        ref_text = ZH_REF.get(Path(rpath).name, "")
    else:
        ref_text = args.ref_text
    t1 = time.time()
    prompt = model.create_voice_clone_prompt(ref_audio=rpath, ref_text=ref_text,
                                             preprocess_prompt=not args.no_preprocess_prompt)
    meta = {"model": "k2-fsa/OmniVoice", "weights_license": "CC-BY-NC（非商用）", "ref_wav": rpath,
            "ref_text_mode": args.ref_text_mode, "ref_text_used": prompt.ref_text,
            "ref_tokens": int(prompt.ref_audio_tokens.shape[-1]), "language": args.language,
            "text_field": args.text_field, "rate": args.rate, "num_step": args.num_step,
            "guidance_scale": args.guidance, "device": args.device, "dtype": str(dtype),
            "load_sec": round(load_sec, 1), "prompt_sec": round(time.time() - t1, 2), "items": []}
    print("ref_text used:", repr(prompt.ref_text), flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in sents:
        text = s[args.text_field]
        dur = (s["syllables"] / args.rate) if args.rate > 0 else None
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t2 = time.time()
            audio = model.generate(text=text, language=args.language, voice_clone_prompt=prompt,
                                   duration=dur, num_step=args.num_step, guidance_scale=args.guidance)
            y = audio[0]
            gen = time.time() - t2
            fn = out_dir / f"s{s['id']}_seed{seed}.wav"
            sf.write(str(fn), y, sr, subtype="PCM_16")
            item = {"id": s["id"], "seed": seed, "text_in": text, "target_dur": round(dur, 2) if dur else None,
                    "file": fn.name, "dur_sec": round(len(y) / sr, 3), "gen_sec": round(gen, 2)}
            meta["items"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    if args.device == "cuda":
        meta["cuda_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        meta["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"all done {time.time()-t0:.1f}s", flush=True)
