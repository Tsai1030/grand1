"""UTMOS22 strong 自然度分數（1–5，越高越自然），當「機器感／金屬感」的粗略指標。
模型：omnivoice 套件內的 UTMOS22Strong 定義＋HF k2-fsa/TTS_eval_models 的 mos/utmos22_strong_step7459_v1.pt。
注意：UTMOS 是用英語（BVCC）訓練的，對台語只能看相對高低，不是絕對分數。
用 envs/omnivoice 的 python、CPU。
用法：python utmos_score.py 檔案或資料夾 ... --json out.json [--recursive]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import librosa  # noqa: E402
import torch  # noqa: E402

torch.accelerator.current_accelerator = lambda *a, **k: None  # 見 gen_omnivoice.py 的說明
from omnivoice.eval.models.utmos import UTMOS22Strong  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "routeB" / "models" / "utmos" / "mos" / "utmos22_strong_step7459_v1.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(6)
    m = UTMOS22Strong()
    m.load_state_dict(torch.load(str(CKPT), map_location="cpu"))
    m.eval()
    files = []
    for p in map(Path, args.inputs):
        files += sorted(p.rglob("*.wav") if args.recursive else p.glob("*.wav")) if p.is_dir() else [p]
    out = {}
    for f in files:
        y, _ = librosa.load(str(f), sr=16000)
        with torch.no_grad():
            s = float(m(torch.from_numpy(y).unsqueeze(0), 16000).item())
        key = str(f.resolve().relative_to(ROOT)) if ROOT in f.resolve().parents else str(f)
        out[key] = round(s, 3)
        print(f"{key}\t{s:.3f}", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"model": "UTMOS22 strong (k2-fsa/TTS_eval_models)", "scores": out},
                                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
