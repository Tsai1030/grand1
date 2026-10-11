import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repos" / "CosyVoice"))
sys.path.insert(0, str(ROOT / "repos" / "Matcha-TTS"))
t = time.time()
from cosyvoice.cli.cosyvoice import AutoModel  # noqa
import torch
print("import ok", round(time.time() - t, 1), "s; torch", torch.__version__, "cuda avail", torch.cuda.is_available())
