"""Fun-CosyVoice3-0.5B-2512 逐句生成（instruct2 模式：指定閩南話＋阿公參考音）。

注意 instruct2 的運作方式（repos/CosyVoice/cosyvoice/cli/frontend.py frontend_instruct2）：
參考音只送進 flow（決定音色／聲學細節），LLM 看不到參考音的 speech token，
所以「節奏、每個字的長短」由 LLM 依指令＋文字決定，跟參考音無關。

用法（在專案根目錄）：
  envs\\cosyvoice\\Scripts\\python.exe routeB\\scripts\\gen_cosyvoice.py --device cuda \
      --prompt ref15=inputs\\voice_reference_15s.wav --prompt clean22=routeA\\sources\\ref_clean_06-08-07.wav \
      --seeds 1 2 3 --variant base
輸出：routeB/cosyvoice/<variant>/<prompt>/s<句號>_seed<k>.wav（24 kHz）＋ 同資料夾 meta.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
ap.add_argument("--prompt", action="append", required=True, help="名稱=音檔路徑")
ap.add_argument("--ids", type=int, nargs="*", help="只做這些句號（預設全部）")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--variant", default="", help="輸出資料夾名稱的前綴（資料夾＝<variant><指令名稱>）")
ap.add_argument("--instructs", nargs="+", default=["base"], help="要跑的指令名稱（見 INSTRUCTS）")
ap.add_argument("--text-field", default="hanzi", help="sentences.json 的欄位：hanzi（辭典用字）或 hanzi_orig（家人原稿）")
ap.add_argument("--text-prefix", default="", help="加在每句前面的字串（例如 [breath]）")
ap.add_argument("--comma-breath", action="store_true", help="把句中的逗號換成「，[breath]」")
ap.add_argument("--llm", choices=["base", "rl"], default="base", help="base=llm.pt；rl=llm.rl.pt")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--threads", type=int, default=6, help="CPU 模式的 torch 執行緒數")
args = ap.parse_args()

# 指令格式照官方 CosyVoice3：「You are a helpful assistant. <指令><|endofprompt|>」
# （repos/CosyVoice/cosyvoice/utils/common.py 的 instruct_list、example.py）。
# 官方清單裡方言、情緒、語速是分開的單一指令；warm／elder 是把「闽南话」和語氣寫在同一句，官方沒有這種範例，屬實驗。
INSTRUCTS = {
    "base": "You are a helpful assistant. 请用闽南话表达。<|endofprompt|>",
    "warm": "You are a helpful assistant. 请用闽南话，用温暖慈祥、像长辈跟家人说话的语气表达。<|endofprompt|>",
    "elder": "You are a helpful assistant. 请用闽南话表达，像老爷爷在跟孙子聊天一样，语气亲切自然、抑扬顿挫。<|endofprompt|>",
}

if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 不碰 GPU（seedvc 在用）

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repos" / "CosyVoice"))
sys.path.insert(0, str(ROOT / "repos" / "Matcha-TTS"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402

if args.device == "cpu":
    torch.set_num_threads(args.threads)
assert (args.device == "cuda") == torch.cuda.is_available(), "device 設定與 CUDA 狀態不符"

MODEL_DIR = ROOT / "routeB" / "models" / "Fun-CosyVoice3-0.5B-2512"
sents = json.loads((ROOT / "routeB" / "sentences.json").read_text(encoding="utf-8"))
if args.ids:
    sents = [s for s in sents if s["id"] in args.ids]

t0 = time.time()
cosy = AutoModel(model_dir=str(MODEL_DIR), fp16=args.fp16)
if args.llm == "rl":
    sd = torch.load(str(MODEL_DIR / "llm.rl.pt"), map_location=cosy.model.device, weights_only=True)
    cosy.model.llm.load_state_dict(sd, strict=True)
    cosy.model.llm.eval()
load_sec = time.time() - t0
sr = cosy.sample_rate
print(f"model loaded in {load_sec:.1f}s, sr={sr}, device={cosy.model.device}", flush=True)


def prompt_cache(wav_path):
    mi = cosy.frontend.frontend_zero_shot("", "", wav_path, sr, "")
    for k in ["text", "text_len", "prompt_text", "prompt_text_len", "llm_prompt_speech_token", "llm_prompt_speech_token_len"]:
        mi.pop(k, None)
    return mi


PROMPTS = {}
for pspec in args.prompt:
    pname, ppath = pspec.split("=", 1)
    ppath = str((ROOT / ppath).resolve()) if not os.path.isabs(ppath) else ppath
    PROMPTS[pname] = (ppath, prompt_cache(ppath))

for iname, pname in [(i, p) for i in args.instructs for p in PROMPTS]:
    instruct = INSTRUCTS[iname]
    instr_tok, instr_len = cosy.frontend._extract_text_token(instruct)
    ppath, pc = PROMPTS[pname]
    out_dir = ROOT / "routeB" / "cosyvoice" / f"{args.variant}{iname}" / pname
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "llm": "llm.rl.pt" if args.llm == "rl" else "llm.pt",
            "mode": "inference_instruct2 (frontend_instruct2 等效：參考音只進 flow)", "instruct_name": iname, "instruct": instruct,
            "prompt_wav": ppath, "device": args.device, "fp16": args.fp16, "speed": args.speed,
            "text_field": args.text_field, "text_prefix": args.text_prefix, "comma_breath": args.comma_breath,
            "load_sec": round(load_sec, 1), "items": []}
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in sents:
        text = s[args.text_field]
        if args.comma_breath:
            text = text.replace("，", "，[breath]")
        text = args.text_prefix + text
        chunks = cosy.frontend.text_normalize(text, split=True, text_frontend=True)
        for seed in args.seeds:
            t1 = time.time()
            pieces = []
            for ch in chunks:
                tt, ttl = cosy.frontend._extract_text_token(ch)
                mi = dict(pc)
                mi.update(text=tt, text_len=ttl, prompt_text=instr_tok, prompt_text_len=instr_len)
                set_all_random_seed(seed)
                for out in cosy.model.tts(**mi, stream=False, speed=args.speed):
                    pieces.append(out["tts_speech"].squeeze(0).cpu().numpy())
            y = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            gen = time.time() - t1
            fn = out_dir / f"s{s['id']}_seed{seed}.wav"
            sf.write(str(fn), y, sr, subtype="PCM_16")
            item = {"id": s["id"], "seed": seed, "text_in": chunks, "file": fn.name,
                    "dur_sec": round(len(y) / sr, 3), "gen_sec": round(gen, 2), "rtf": round(gen / max(len(y) / sr, 1e-3), 3)}
            meta["items"].append(item)
            print(iname, pname, json.dumps(item, ensure_ascii=False), flush=True)
    if args.device == "cuda":
        meta["cuda_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        meta["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"all done {time.time()-t0:.1f}s", flush=True)
