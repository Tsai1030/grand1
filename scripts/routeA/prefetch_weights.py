"""預先下載 Seed-VC v1（whisper-small-wavenet）與 v2（hubert-bsqvae-small）推論與微調會用到的權重。

放的位置與 repo 程式碼自己下載時一致：
  - repos/seed-vc/checkpoints/            （hf_utils.load_custom_model_from_hf 的 cache_dir）
  - repos/seed-vc/checkpoints/hf_cache/   （transformers / BigVGAN；我們的包裝腳本把 HF_HUB_CACHE 指到這裡）
不需要 HF token（全部是公開、非 gated 模型）。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "repos", "seed-vc"))
os.environ["HF_HUB_CACHE"] = os.path.join(REPO, "checkpoints", "hf_cache")
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files  # noqa: E402

CK = os.path.join(REPO, "checkpoints")

FILES = [
    # v1 推論（f0-condition False 的預設模型）＋微調起點
    ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
    ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
    # v2
    ("Plachta/Seed-VC", "v2/cfm_small.pth"),
    ("Plachta/Seed-VC", "v2/ar_base.pth"),
    ("Plachta/ASTRAL-quantization", "bsq32/bsq32_light.pth"),
    ("Plachta/ASTRAL-quantization", "bsq2048/bsq2048_light.pth"),
    # 共用：聲紋（style）編碼器
    ("funasr/campplus", "campplus_cn_common.bin"),
    # v1 微調才用：OpenVoice 音色擾動器與語者向量庫
    ("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth"),
    ("myshell-ai/OpenVoiceV2", "converter/config.json"),
    ("Plachta/Seed-VC", "se_db.pt"),
]

SNAPSHOTS = [
    ("openai/whisper-small", ["*.json", "*.txt", "model.safetensors"]),
    ("nvidia/bigvgan_v2_22khz_80band_256x", ["config.json", "bigvgan_generator.pt"]),
]


def main():
    only = sys.argv[1:]  # 可只下載部分（比對 repo 名稱）
    for repo, fn in FILES:
        if only and not any(o in repo or o in fn for o in only):
            continue
        t = time.time()
        p = hf_hub_download(repo_id=repo, filename=fn, cache_dir=CK)
        print(f"[ok {time.time()-t:6.1f}s] {repo}/{fn} -> {os.path.getsize(p)/2**20:.1f} MiB", flush=True)
    hub = [("facebook/hubert-large-ll60k", None)] + SNAPSHOTS
    for repo, pats in hub:
        if only and not any(o in repo for o in only):
            continue
        if pats is None:
            files = list_repo_files(repo)
            w = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
            pats = ["config.json", "preprocessor_config.json", w]
        t = time.time()
        p = snapshot_download(repo_id=repo, allow_patterns=pats)
        print(f"[ok {time.time()-t:6.1f}s] {repo} {pats} -> {p}", flush=True)
    print("PREFETCH_DONE")


if __name__ == "__main__":
    main()
