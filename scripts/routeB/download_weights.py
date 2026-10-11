"""下載路線 B 用的模型權重（Hugging Face，不需 token）。
用法：python download_weights.py cosyvoice|omnivoice
存到 routeB/models/<名稱>，並記錄 commit。
"""
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "routeB" / "models"

JOBS = {
    "cosyvoice": dict(
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local="Fun-CosyVoice3-0.5B-2512",
        # 不下載：flow.decoder.estimator.fp32.onnx（TensorRT 用）、speech_tokenizer_v3.batch.onnx（訓練批次用）、asset/
        allow=[
            "CosyVoice-BlankEN/*",
            "campplus.onnx",
            "config.json",
            "configuration.json",
            "cosyvoice3.yaml",
            "README.md",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "llm.rl.pt",
            "speech_tokenizer_v3.onnx",
        ],
    ),
    "omnivoice": dict(repo_id="k2-fsa/OmniVoice", local="OmniVoice", allow=None),
}

name = sys.argv[1]
job = JOBS[name]
out = MODELS / job["local"]
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
path = snapshot_download(
    repo_id=job["repo_id"],
    local_dir=str(out),
    allow_patterns=job["allow"],
    max_workers=8,
)
from huggingface_hub import HfApi

sha = HfApi().model_info(job["repo_id"]).sha
(out / "_SOURCE.txt").write_text(f"{job['repo_id']} @ {sha}\n", encoding="utf-8")
print(f"done {name} -> {path}  sha={sha}  {time.time()-t0:.0f}s", flush=True)
