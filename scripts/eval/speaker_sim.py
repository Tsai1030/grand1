"""聲紋相似度（3D-Speaker CAM++ / ERes2NetV2，餘弦相似度）。

例：
  # 測試音（檔案或資料夾）分別跟阿公原始錄音前半段、後半段各比一次（計畫書「評估方式」的做法）
  envs\\prep\\Scripts\\python.exe eval\\speaker_sim.py --agong-halves --test routeA\\out\\ --json routeA\\sim.json

  # 自訂多個參考與測試
  envs\\prep\\Scripts\\python.exe eval\\speaker_sim.py --ref inputs\\voice_reference_15s.wav stage1\\segments --test a.wav b.wav

輸出：表格（列＝測試音、欄＝各參考，最後一欄是對所有參考的平均）；--json 另存完整結果。
模型：預設 iic/speech_campplus_sv_zh-cn_16k-common（CAM++，20 萬中文語者訓練，modelscope 不需授權）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_io import ROOT, AudioItem, expand_paths, halves  # noqa: E402

AGONG_RAW_WAV = ROOT / "stage1" / "wav" / "male_voice_only.wav"
AGONG_RAW_MP3 = ROOT / "inputs" / "male_voice_only.mp3"
AGONG_DENOISE_WAV = ROOT / "stage1" / "enhanced" / "male_voice_only_denoise.wav"
SPEAKERLAB_REPO = ROOT / "repos" / "3D-Speaker"
MODEL_CACHE = ROOT / "eval" / "models"

MODELS = {
    "campplus": {
        "model_id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "revision": "v1.0.0",
        "model_pt": "campplus_cn_common.bin",
        "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
        "args": {"feat_dim": 80, "embedding_size": 192},
        # 同一份權重也放在 HF 的 funasr/campplus（modelscope 連不上時的備援）
        "hf_fallback": ("funasr/campplus", "campplus_cn_common.bin"),
    },
    "eres2netv2": {
        "model_id": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "revision": "v1.0.1",
        "model_pt": "pretrained_eres2netv2.ckpt",
        "obj": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
        "args": {"feat_dim": 80, "embedding_size": 192, "baseWidth": 26, "scale": 2, "expansion": 2},
        "hf_fallback": None,
    },
}

SR = 16000


class SpeakerEmbedder:
    def __init__(self, model: str = "campplus", device: str = "auto"):
        import importlib
        import torch

        self.torch = torch
        conf = MODELS[model]
        self.conf = conf
        self.model_name = model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = self._fetch_weights(conf)
        sys.path.insert(0, str(SPEAKERLAB_REPO))
        mod_name, cls_name = conf["obj"].rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        net = cls(**conf["args"])
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        self.net = net.to(self.device).eval()
        self.ckpt = str(ckpt)
        self._cache: dict[tuple, np.ndarray] = {}

    @staticmethod
    def _fetch_weights(conf) -> Path:
        local = MODEL_CACHE / conf["model_id"].split("/")[1] / conf["model_pt"]
        if local.exists():
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        err = None
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            d = snapshot_download(conf["model_id"], revision=conf["revision"],
                                  cache_dir=str(MODEL_CACHE / "_modelscope"))
            src = Path(d) / conf["model_pt"]
            if src.exists():
                local.write_bytes(src.read_bytes())
                return local
            err = f"{src} 不存在"
        except Exception as e:  # noqa: BLE001
            err = e
        if conf.get("hf_fallback"):
            from huggingface_hub import hf_hub_download
            repo, fn = conf["hf_fallback"]
            print(f"[speaker_sim] modelscope 下載失敗（{err}），改用 HF {repo}", file=sys.stderr)
            p = hf_hub_download(repo, fn, cache_dir=str(MODEL_CACHE / "_hf"))
            local.write_bytes(Path(p).read_bytes())
            return local
        raise RuntimeError(f"下載 {conf['model_id']} 失敗：{err}")

    def embed_array(self, y: np.ndarray) -> np.ndarray:
        import torchaudio.compliance.kaldi as kaldi
        torch = self.torch
        wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
        # 與 3D-Speaker speakerlab FBank(80, 16000, mean_nor=True) 相同
        feat = kaldi.fbank(wav, num_mel_bins=80, sample_frequency=SR, dither=0)
        feat = feat - feat.mean(0, keepdim=True)
        with torch.no_grad():
            emb = self.net(feat.unsqueeze(0).to(self.device)).squeeze(0).float().cpu().numpy()
        return emb

    def embed(self, item: AudioItem) -> np.ndarray:
        key = (str(Path(item.path).resolve()), item.start, item.end)
        if key not in self._cache:
            y = item.load(SR)
            if len(y) < SR * 0.5:
                print(f"[speaker_sim] 警告：{item.name} 只有 {len(y) / SR:.2f} 秒，分數不可靠", file=sys.stderr)
            self._cache[key] = self.embed_array(y)
        return self._cache[key]

    def info(self) -> dict:
        return {"model": self.model_name, "model_id": self.conf["model_id"],
                "revision": self.conf["revision"], "checkpoint": self.ckpt,
                "sample_rate": SR, "feature": "kaldi fbank 80, mean-normalized",
                "score": "cosine similarity", "device": str(self.device)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def items_from(paths: list[str], recursive: bool) -> list[AudioItem]:
    files = expand_paths(paths, recursive)
    names = [f.name for f in files]
    out = []
    for f in files:
        # 同名檔案（不同資料夾）時加上上一層資料夾名稱避免混淆
        name = f.name if names.count(f.name) == 1 else f"{f.parent.name}/{f.name}"
        out.append(AudioItem(name, f))
    return out


def print_table(rows: list[str], cols: list[str], mat: dict, mean_col: dict | None):
    w0 = max([len(r) for r in rows] + [6])
    hdr = ["測試＼參考".ljust(w0)] + [c for c in cols] + (["平均"] if mean_col else [])
    print("\t".join(hdr))
    for r in rows:
        vals = [f"{mat[r][c]:.4f}" for c in cols]
        if mean_col:
            vals.append(f"{mean_col[r]:.4f}")
        print("\t".join([r.ljust(w0)] + vals))


def main(argv=None):
    ap = argparse.ArgumentParser(description="聲紋餘弦相似度（3D-Speaker）")
    ap.add_argument("--ref", nargs="*", default=[], help="參考音：檔案或資料夾（可多個）")
    ap.add_argument("--test", nargs="*", default=[], help="測試音：檔案或資料夾（可多個）")
    ap.add_argument("--agong-halves", action="store_true",
                    help="加入阿公原始錄音（stage1/wav/male_voice_only.wav）的前半、後半當參考")
    ap.add_argument("--agong-denoise-halves", action="store_true",
                    help="加入阿公只去噪版（stage1/enhanced/male_voice_only_denoise.wav）的前半、後半當參考"
                         "（切點與原始錄音相同）")
    ap.add_argument("--halves", nargs="*", default=[],
                    help="把指定檔案切成前半、後半（切在中點附近最安靜處）當參考")
    ap.add_argument("--model", choices=sorted(MODELS), default="campplus")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--recursive", action="store_true", help="資料夾遞迴搜尋")
    ap.add_argument("--json", type=Path, help="結果另存 JSON")
    args = ap.parse_args(argv)

    refs: list[AudioItem] = []
    agong_split = None
    if args.agong_halves or args.agong_denoise_halves:
        src = AGONG_RAW_WAV if AGONG_RAW_WAV.exists() else AGONG_RAW_MP3
        orig_halves = halves(src, "阿公原始")
        agong_split = orig_halves[0].end
        if args.agong_halves:
            refs += orig_halves
    if args.agong_denoise_halves:
        refs += halves(AGONG_DENOISE_WAV, "阿公去噪", split=agong_split)
    for h in args.halves:
        refs += halves(Path(h))
    refs += items_from(args.ref, args.recursive)
    tests = items_from(args.test, args.recursive)
    if not refs:
        ap.error("至少要有一個參考（--ref / --agong-halves / --agong-denoise-halves / --halves）")

    emb = SpeakerEmbedder(args.model, args.device)
    pairwise = not tests
    if pairwise:  # 沒給測試音：參考兩兩互比
        tests = refs

    ref_e = {r.name: emb.embed(r) for r in refs}
    scores: dict[str, dict[str, float]] = {}
    mean_over: dict[str, float] = {}
    for t in tests:
        te = emb.embed(t)
        scores[t.name] = {r.name: cosine(te, ref_e[r.name]) for r in refs}
        others = [v for k, v in scores[t.name].items() if not (pairwise and k == t.name)]
        mean_over[t.name] = float(np.mean(others)) if others else float("nan")

    print(f"# 模型：{emb.conf['model_id']}（{args.model}），餘弦相似度")
    for r in refs:
        if r.start is not None:
            print(f"# 參考 {r.name}：{r.path} {r.start:.2f}–{r.end:.2f} 秒")
    print_table([t.name for t in tests], [r.name for r in refs], scores, mean_over)

    if args.json:
        out = {"info": emb.info(), "refs": [r.describe() for r in refs],
               "tests": [t.describe() for t in tests], "scores": scores,
               "mean_over_refs": mean_over, "pairwise": pairwise}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"# JSON 已存：{args.json}")
    return scores


if __name__ == "__main__":
    main()
