"""Breeze-ASR-26 把候選句轉成華語，對照每句的華語意思，抓出唸錯或聽不懂的句子。

用 envs/prep 的 python。設定跟 stage1/scripts/run_asr.py 相同（language=zh、transcribe、beam 5、
壓縮比 > 2.4 視為重複幻覺→加重複懲罰重跑）。
檔名要是 s<句號>_*.wav（逐句檔），用句號對照 routeB/sentences.json。

判定（自動、粗略）：每句列出幾組「關鍵意思」（同義詞任一出現即算命中），
命中率 = 命中組數 / 組數。命中率 < 0.75 → 標「有問題」。最後一定要人工看 ASR 文字。

用法：python asr_check.py 檔案或資料夾 ... --json out.json [--device cuda|cpu] [--recursive]
"""
import argparse
import difflib
import json
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from audio_io import load_audio  # noqa: E402

MODEL_ID = "MediaTek-Research/Breeze-ASR-26"

KEYS = {
    1: [["我是", "我就是", "我係"], ["阿公", "爺爺", "阿爺", "公公", "祖父"]],
    2: [["這裡", "這邊", "這兒"], ["一切"], ["好"], ["不用", "不必", "別", "不要", "免"], ["煩惱", "擔心", "操心"]],
    3: [["謝謝", "感謝", "多謝"], ["這幾年", "幾年", "這些年", "多年"], ["照顧"], ["陪伴", "陪"]],
    4: [["希望"], ["以後", "將來", "往後"], ["日子"], ["每天", "天天", "每一天"], ["開心", "快樂", "歡喜", "高興"]],
    5: [["身體"], ["照顧", "顧好", "保重"], ["吃飽", "吃"], ["睡"]],
    6: [["一家", "全家", "家人", "一家人"], ["疼", "愛惜", "互相", "珍惜"], ["事情", "事"], ["幫忙", "幫助", "互助", "幫"]],
    7: [["天上", "天頂", "天堂"], ["保佑", "保祐", "保庇", "庇佑", "庇祐", "保護"], ["平安"]],
    8: [["疼", "愛"], ["你們", "你"]],
}


def score(sid, text):
    groups = KEYS.get(sid, [])
    hit = [any(k in text for k in g) for g in groups]
    return (sum(hit) / len(groups) if groups else None), [g[0] for g, h in zip(groups, hit) if not h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    sents = {s["id"]: s for s in json.loads((ROOT / "routeB" / "sentences.json").read_text(encoding="utf-8"))}
    files = []
    for p in args.inputs:
        p = Path(p)
        if not p.exists():
            print(f"略過（不存在）：{p}", flush=True)
            continue
        if p.is_dir():
            files += sorted(p.rglob("s*.wav") if args.recursive else p.glob("s*.wav"))
        else:
            files.append(p)

    from transformers import AutoProcessor, WhisperForConditionalGeneration
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(args.device).eval()
    print(f"loaded in {time.time() - t0:.1f}s on {args.device}", flush=True)

    rows = []
    for f in files:
        m = re.match(r"s(\d+)_", f.name)
        sid = int(m.group(1)) if m else None
        y, _ = load_audio(f, 16000)
        feats = proc(y.astype(np.float32), sampling_rate=16000, return_tensors="pt").input_features.to(args.device, dtype)
        mask = torch.ones(feats.shape[0], feats.shape[-1], dtype=torch.long, device=args.device)

        def run(**kw):
            with torch.inference_mode():
                ids = model.generate(feats, attention_mask=mask, language="zh", task="transcribe",
                                     num_beams=args.beams, max_new_tokens=120, **kw)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

        text = run()
        if len(text) > 20 and len(text.encode()) / len(zlib.compress(text.encode())) > 2.4:
            text = run(no_repeat_ngram_size=4, repetition_penalty=1.2)
        exp = sents[sid]["zh"] if sid in sents else ""
        sc, miss = score(sid, text) if sid else (None, [])
        sim = difflib.SequenceMatcher(None, re.sub(r"\W", "", exp), re.sub(r"\W", "", text)).ratio() if exp else None
        row = {"file": str(f.relative_to(ROOT)) if f.is_absolute() and ROOT in f.parents else str(f),
               "id": sid, "asr": text, "expected": exp, "key_hit": sc, "missing": miss,
               "char_sim": round(sim, 3) if sim is not None else None,
               "problem": (sc is not None and sc < 0.75)}
        rows.append(row)
        print(f"{row['file']}\t{sc if sc is None else round(sc,2)}\t{'有問題' if row['problem'] else 'OK'}\t{text}", flush=True)

    meta = {"model": MODEL_ID, "device": args.device, "dtype": str(dtype), "language": "zh", "task": "transcribe",
            "num_beams": args.beams, "rule": "key_hit < 0.75 → problem", "keys": KEYS}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.device == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
