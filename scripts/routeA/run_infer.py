"""Seed-VC 推論包裝：固定亂數種子、記錄時間與顯存，輸出改成指定檔名。

在 repos/seed-vc 目錄下執行原本的 inference.py / inference_v2.py（runpy），不改 repo 原始碼。

用法：
  python run_infer.py --model v1 --out path\\to\\out.wav [--seed 1234] [--cpu] -- <inference.py 參數>
  python run_infer.py --model v2 --out path\\to\\out.wav -- <inference_v2.py 參數>
（--source/--target 請給絕對路徑；--output 不用給，會自動用暫存資料夾。）

結果另外寫一行 JSON 到 <out>.json：秒數、顯存峰值等。
"""
import argparse
import glob
import json
import os
import random
import runpy
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.join(ROOT, "repos", "seed-vc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v1", "v2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true", help="強制用 CPU（GPU 規則未解除前的環境測試）")
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own, rest = argv[:i], argv[i + 1:]
    else:
        own, rest = argv, []
    args = ap.parse_args(own)
    out = os.path.abspath(args.out)

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_CACHE"] = os.path.join(REPO, "checkpoints", "hf_cache")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.chdir(REPO)
    sys.path.insert(0, REPO)

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu:
        # CPU 不支援 Whisper 的 fp16 卷積：強制 float32
        import transformers
        orig = transformers.WhisperModel.from_pretrained.__func__

        def fp32_from_pretrained(cls, *a, **kw):
            kw["torch_dtype"] = torch.float32
            return orig(cls, *a, **kw)

        transformers.WhisperModel.from_pretrained = classmethod(fp32_from_pretrained)

    tmpdir = tempfile.mkdtemp(prefix="seedvc_out_")
    script = "inference.py" if args.model == "v1" else "inference_v2.py"
    sys.argv = [script] + rest + ["--output", tmpdir]
    t0 = time.time()
    runpy.run_path(os.path.join(REPO, script), run_name="__main__")
    wall = time.time() - t0

    wavs = sorted(glob.glob(os.path.join(tmpdir, "*.wav")), key=os.path.getmtime)
    if not wavs:
        raise SystemExit("沒有產生輸出檔")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.move(wavs[-1], out)
    shutil.rmtree(tmpdir, ignore_errors=True)

    info = {
        "model": args.model,
        "out": out,
        "seed": args.seed,
        "args": rest,
        "wall_sec_including_model_load": round(wall, 2),
        "device": "cpu" if args.cpu or not torch.cuda.is_available() else torch.cuda.get_device_name(0),
    }
    if torch.cuda.is_available():
        info["torch_max_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        info["torch_max_reserved_GB"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    import soundfile as sf
    d = sf.info(out)
    info["out_sec"] = round(d.frames / d.samplerate, 2)
    info["out_sr"] = d.samplerate
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("RUN_INFER_RESULT " + json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
