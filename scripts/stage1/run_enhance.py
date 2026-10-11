"""階段一：用 resemble-enhance 對阿公錄音做「只去噪」與「完整增強」。

Windows 上 pip 版 resemble-enhance 依賴 deepspeed（訓練用）裝不起來，
這裡直接用 clone 下來的 repo（repos/resemble-enhance）的 Python API，
並在 import 前塞一個只供 import 用的 deepspeed 假模組（推論完全不會呼叫到）。

用法（envs/prep 的 python）：
  python stage1/scripts/run_enhance.py --in stage1/wav/male_voice_only.wav --out-dir stage1/enhanced
"""
import argparse
import functools
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RE_REPO = ROOT / "repos" / "resemble-enhance"


def _install_deepspeed_stub():
    """resemble_enhance 在 import 時會 import deepspeed（只在訓練用到）。"""
    if "deepspeed" in sys.modules:
        return
    ds = types.ModuleType("deepspeed")
    ds.DeepSpeedConfig = lambda *a, **k: None
    ds.init_distributed = lambda *a, **k: None
    acc = types.ModuleType("deepspeed.accelerator")
    acc.get_accelerator = lambda: None
    rt = types.ModuleType("deepspeed.runtime")
    eng = types.ModuleType("deepspeed.runtime.engine")
    eng.DeepSpeedEngine = type("DeepSpeedEngine", (object,), {})
    ut = types.ModuleType("deepspeed.runtime.utils")
    ut.clip_grad_norm_ = lambda *a, **k: None
    ds.accelerator, ds.runtime, rt.engine, rt.utils = acc, rt, eng, ut
    sys.modules.update({
        "deepspeed": ds,
        "deepspeed.accelerator": acc,
        "deepspeed.runtime": rt,
        "deepspeed.runtime.engine": eng,
        "deepspeed.runtime.utils": ut,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--nfe", type=int, default=64)          # 同 resemble-enhance CLI 預設
    ap.add_argument("--solver", default="midpoint")         # 同 CLI 預設
    ap.add_argument("--lambd", type=float, default=1.0)     # 同 CLI 預設（去噪強度）
    ap.add_argument("--tau", type=float, default=0.5)       # 同 CLI 預設
    ap.add_argument("--extra-lambd", type=float, nargs="*", default=[0.5],
                    help="另外再跑的較輕 lambd 版本（只供比較）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk-seconds", type=float, default=10.0, help="原版 30；8GB 卡用 10")
    args = ap.parse_args()

    _install_deepspeed_stub()
    sys.path.insert(0, str(RE_REPO))

    import torch
    import soundfile as sf

    # 官方 checkpoint 是 deepspeed 格式（含非 tensor 物件），torch 2.6 預設 weights_only=True 會拒載。
    # 來源是官方 HF repo ResembleAI/resemble-enhance，這裡放寬為 weights_only=False。
    torch.load = functools.partial(torch.load, weights_only=False)

    # 官方 hparams.yaml 存了 !!python/object/apply:pathlib.PosixPath，Windows 無法建立 PosixPath。
    if sys.platform == "win32":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

    import resemble_enhance.enhancer.inference as rei
    from resemble_enhance.enhancer.inference import denoise, enhance

    # 原版每塊 30 秒；8GB 卡（桌面已佔 1–2GB）跑完整增強會爆顯存、掉到共享記憶體變超慢，改小塊。
    rei.inference = functools.partial(rei.inference, chunk_seconds=args.chunk_seconds, overlap_seconds=1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sr = sf.read(str(args.inp), dtype="float32", always_2d=True)
    dwav = torch.from_numpy(wav.mean(1))
    stem = args.inp.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def save(hwav, out_sr, name):
        p = args.out_dir / name
        sf.write(str(p), hwav.cpu().numpy(), out_sr, subtype="FLOAT")
        print(f"saved {p} sr={out_sr} len={hwav.shape[-1] / out_sr:.2f}s")

    import time
    if not (args.out_dir / f"{stem}_denoise.wav").exists():
        hwav, out_sr = denoise(dwav, sr, device)
        save(hwav, out_sr, f"{stem}_denoise.wav")

    runs = [(args.lambd, f"{stem}_enhance.wav")]
    for lam in args.extra_lambd:
        runs.append((lam, f"{stem}_enhance_lambd{lam:g}.wav"))
    for lam, name in runs:
        t0 = time.time()
        torch.manual_seed(args.seed)
        hwav, out_sr = enhance(dwav, sr, device, nfe=args.nfe, solver=args.solver, lambd=lam, tau=args.tau)
        save(hwav, out_sr, name)
        if device == "cuda":
            print(f"  {time.time() - t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)


if __name__ == "__main__":
    main()
