"""stage1 腳本共用：路徑、VAD、雜訊估計。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE1 = ROOT / "stage1"
sys.path.insert(0, str(ROOT / "eval"))

ORIG_WAV = STAGE1 / "wav" / "male_voice_only.wav"
ENH_DIR = STAGE1 / "enhanced"
VERSIONS = {
    "original": ORIG_WAV,
    "denoise": ENH_DIR / "male_voice_only_denoise.wav",
    "enhance": ENH_DIR / "male_voice_only_enhance.wav",
    "enhance_lambd0.5": ENH_DIR / "male_voice_only_enhance_lambd0.5.wav",
}

VAD_PARAMS = dict(threshold=0.5, min_silence_duration_ms=150, min_speech_duration_ms=150, speech_pad_ms=0)


def vad_regions(y16: np.ndarray) -> list[tuple[float, float]]:
    """silero-vad（16 kHz）→ [(start_sec, end_sec), ...]"""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(y16), model, sampling_rate=16000, **VAD_PARAMS)
    return [(t["start"] / 16000, t["end"] / 16000) for t in ts]


def locate_clip(full: np.ndarray, clip: np.ndarray, sr: int) -> tuple[float, float]:
    """用 FFT 互相關找 clip 在 full 裡的位置 → (offset_sec, normalized_corr)。"""
    from scipy.signal import fftconvolve
    corr = fftconvolve(full, clip[::-1], mode="valid")
    k = int(np.argmax(corr))
    seg = full[k:k + len(clip)]
    nc = float(np.dot(seg, clip) / (np.linalg.norm(seg) * np.linalg.norm(clip) + 1e-12))
    return k / sr, nc


def frame_db(y: np.ndarray, sr: int, win: float = 0.05, hop: float = 0.01) -> np.ndarray:
    w, h = int(win * sr), int(hop * sr)
    n = max(1, 1 + (len(y) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    fr = y[np.minimum(idx, len(y) - 1)]
    return 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)


def noise_stats(y: np.ndarray, sr: int, regions: list[tuple[float, float]], guard: float = 0.1) -> dict:
    """以 VAD 區段把 y 分成語音／非語音（邊界各留 guard 秒不算），算 RMS dBFS 與粗估 SNR。"""
    db = frame_db(y, sr)
    t = np.arange(len(db)) * 0.01 + 0.025
    speech = np.zeros(len(db), bool)
    near = np.zeros(len(db), bool)
    for s, e in regions:
        speech |= (t >= s) & (t <= e)
        near |= (t >= s - guard) & (t <= e + guard)
    nonspeech = ~near
    p = lambda m: 10 * np.log10(np.mean(10 ** (db[m] / 10))) if m.any() else float("nan")  # noqa: E731
    sp, ns = p(speech), p(nonspeech)
    return {
        "speech_rms_dbfs": round(float(sp), 2),
        "nonspeech_rms_dbfs": round(float(ns), 2),
        "snr_est_db": round(float(sp - ns), 2),
        "nonspeech_p50_dbfs": round(float(np.median(db[nonspeech])), 2) if nonspeech.any() else None,
        "nonspeech_sec": round(float(nonspeech.sum() * 0.01), 2),
    }
