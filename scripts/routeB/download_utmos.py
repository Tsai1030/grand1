"""下載 UTMOS22 strong（自然度 MOS 預測器）權重：HF k2-fsa/TTS_eval_models 的 mos/utmos22_strong_step7459_v1.pt。"""
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "routeB" / "models" / "utmos"
out.mkdir(parents=True, exist_ok=True)
p = hf_hub_download("k2-fsa/TTS_eval_models", "mos/utmos22_strong_step7459_v1.pt", local_dir=str(out))
sha = HfApi().model_info("k2-fsa/TTS_eval_models").sha
(out / "_SOURCE.txt").write_text(f"k2-fsa/TTS_eval_models @ {sha} : mos/utmos22_strong_step7459_v1.pt\n", encoding="utf-8")
print(p)
