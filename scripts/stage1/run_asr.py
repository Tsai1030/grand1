"""步驟 12：Breeze-ASR-26（台語語音 → 華語文字）幫每段產出「華語意思」。

模型：MediaTek-Research/Breeze-ASR-26（Whisper-large-v2 微調，Apache-2.0），fp16、GPU。
model card 沒附程式範例，這裡用 transformers 標準 Whisper 用法：language="zh"、task="transcribe"、beam 5。
輸出 stage1/asr.json：{檔名: 文字}
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from common import STAGE1
from audio_io import load_audio

MODEL_ID = "MediaTek-Research/Breeze-ASR-26"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[str(STAGE1 / "segments"), str(STAGE1 / "rejected" / "segments")])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--beams", type=int, default=5)
    args = ap.parse_args()

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(args.device).eval()
    print(f"loaded in {time.time() - t0:.1f}s on {args.device} {dtype}", flush=True)

    out, retry_log = {}, {}
    for d in args.dirs:
        for f in sorted(Path(d).glob("*.wav")):
            y, _ = load_audio(f, 16000)
            feats = proc(y.astype(np.float32), sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(args.device, dtype)
            mask = torch.ones(feats.shape[0], feats.shape[-1], dtype=torch.long, device=args.device)

            def run(**kw):
                with torch.inference_mode():
                    ids = model.generate(feats, attention_mask=mask, language="zh", task="transcribe",
                                         num_beams=args.beams, max_new_tokens=220, **kw)
                return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

            text = run()
            # Whisper 偶爾會無限重複同一句（幻覺）；壓縮比 > 2.4 時加重複懲罰重跑
            if len(text) > 20 and len(text.encode()) / len(zlib.compress(text.encode())) > 2.4:
                retried = run(no_repeat_ngram_size=4, repetition_penalty=1.2)
                print(f"  [重複幻覺，重跑] {text[:40]}… → {retried}", flush=True)
                retry_log[f.name] = {"first": text, "retried": retried}
                text = retried
            out[f.name] = text
            print(f"{f.name}\t{text}", flush=True)
    meta = {"model": MODEL_ID, "device": args.device, "dtype": str(dtype), "language": "zh",
            "task": "transcribe", "num_beams": args.beams,
            "transformers": __import__("transformers").__version__,
            "repetition_retry": "compression ratio > 2.4 → no_repeat_ngram_size=4, repetition_penalty=1.2"}
    (STAGE1 / "asr.json").write_text(json.dumps({"meta": meta, "text": out, "retried": retry_log},
                                                ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    if args.device == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
