"""Seed-VC 微調包裝（v1: train.py；v2: train_v2.py），不改 repo 原始碼。

原本的訓練程式只保留最後 1–2 個存檔點，這裡讓「找舊存檔點」的 glob 回傳空清單，
所以每 100 步的存檔點都會留下來（代價：不能中途續訓）。

另外記錄：
  - 每次存檔的時間（相對開始秒數）
  - torch 顯存峰值（max_memory_allocated / reserved）
  - nvidia-smi 整張卡的顯存峰值（含桌面約 1.2GB）
寫到 <run_dir>/train_stats.json。

用法（在任何目錄）：
  python train_keepall.py --model v1 --run-dir <abs dir> -- --config <yml> --dataset-dir <dir> --batch-size 2 --max-steps 500 --save-every 100 --num-workers 0
  python train_keepall.py --model v2 --run-dir <abs dir> -- --dataset-dir <dir> --batch-size 2 --max-steps 500 --save-every 100 --num-workers 0 --train-cfm
"""
import argparse
import atexit
import glob as _glob_mod
import json
import os
import random
import runpy
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.join(ROOT, "repos", "seed-vc")

stats = {"saves": [], "smi_peak_used_MiB": 0, "smi_baseline_used_MiB": None}
_stop = threading.Event()


def smi_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def poll():
    while not _stop.is_set():
        u = smi_used()
        if u is not None and u > stats["smi_peak_used_MiB"]:
            stats["smi_peak_used_MiB"] = u
        _stop.wait(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v1", "v2"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    argv = sys.argv[1:]
    i = argv.index("--")
    args = ap.parse_args(argv[:i])
    rest = argv[i + 1:]
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    os.environ["HF_HUB_CACHE"] = os.path.join(REPO, "checkpoints", "hf_cache")
    os.chdir(REPO)
    sys.path.insert(0, REPO)

    # 保留所有存檔點：只攔截找存檔點的樣式
    orig_glob = _glob_mod.glob

    def glob_keepall(pattern, *a, **kw):
        base = os.path.basename(pattern)
        if base.startswith(("DiT_epoch_", "CFM_epoch_", "AR_epoch_")):
            return []
        return orig_glob(pattern, *a, **kw)

    _glob_mod.glob = glob_keepall

    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    stats["smi_baseline_used_MiB"] = smi_used()
    stats["start"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["model"] = args.model
    stats["args"] = rest
    th = threading.Thread(target=poll, daemon=True)
    th.start()

    orig_save = torch.save

    def save_logged(obj, f, *a, **kw):
        r = orig_save(obj, f, *a, **kw)
        if isinstance(f, str):
            el = round(time.time() - t0, 1)
            stats["saves"].append({"file": os.path.basename(f), "elapsed_sec": el})
            print(f"[keepall] saved {os.path.basename(f)} at {el}s", flush=True)
        return r

    torch.save = save_logged

    def finish():
        _stop.set()
        stats["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        stats["total_sec"] = round(time.time() - t0, 1)
        if torch.cuda.is_available():
            stats["torch_max_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            stats["torch_max_reserved_GB"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
            stats["gpu"] = torch.cuda.get_device_name(0)
        with open(os.path.join(run_dir, "train_stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=1)
        print("TRAIN_STATS " + json.dumps(stats, ensure_ascii=False), flush=True)

    atexit.register(finish)

    script = "train.py" if args.model == "v1" else "train_v2.py"
    sys.argv = [script] + rest + ["--run-name", run_dir]
    runpy.run_path(os.path.join(REPO, script), run_name="__main__")


if __name__ == "__main__":
    main()
