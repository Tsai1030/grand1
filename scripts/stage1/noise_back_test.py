"""診斷：只去噪版聲紋分數下降，是音色被改，還是只因為雜音被拿掉？把原始錄音的背景雜音加回去版再比一次。輸出 stage1/noise_back_test.json"""
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import VERSIONS, ORIG_WAV, vad_regions  # noqa
from audio_io import load_audio, quiet_split_point  # noqa
from speaker_sim import SpeakerEmbedder, cosine  # noqa

yo, _ = load_audio(ORIG_WAV, 16000)
yd, _ = load_audio(VERSIONS["denoise"], 16000)
n = min(len(yo), len(yd)); yo, yd = yo[:n], yd[:n]
regs = vad_regions(yd)
mask = np.ones(n, bool)
for s, e in regs:
    mask[max(0, int((s - 0.15) * 16000)):int((e + 0.15) * 16000)] = False
mask &= np.abs(yo) > 1e-5  # exclude zeroed parts
noise = yo[mask]
print("noise sec", len(noise) / 16000, "rms dB", 20 * np.log10(np.sqrt(np.mean(noise ** 2))))
rng = np.random.default_rng(0)
reps = int(np.ceil(n / len(noise)))
nz = np.tile(noise, reps)[:n]
t = quiet_split_point(yo, 16000); k = int(t * 16000)
e = SpeakerEmbedder("campplus", "cpu")
O1, O2 = e.embed_array(yo[:k]), e.embed_array(yo[k:])
res = {}
for g_db in (-100, -6, 0, 6):
    y = yd + nz * 10 ** (g_db / 20)
    D1, D2 = e.embed_array(y[:k]), e.embed_array(y[k:])
    print(f"noise gain {g_db:+} dB: cross-half {np.mean([cosine(D1, O2), cosine(D2, O1)]):.4f} same {np.mean([cosine(D1, O1), cosine(D2, O2)]):.4f}")
    res[f"{g_db:+}dB"] = {"cross_half": round(float(np.mean([cosine(D1, O2), cosine(D2, O1)])), 4), "same_content": round(float(np.mean([cosine(D1, O1), cosine(D2, O2)])), 4)}
import json
(__import__("pathlib").Path(__file__).resolve().parents[1] / "noise_back_test.json").write_text(json.dumps({"noise_source": "原始錄音非語音、非靜音區段（約 5.2 秒）循環鋪滿", "model": "campplus", "original_cross_half": round(float(cosine(O1, O2)), 4), "denoise_plus_noise": res}, ensure_ascii=False, indent=2), encoding="utf-8")
