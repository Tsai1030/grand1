"""共用的音檔讀取工具（speaker_sim.py、prosody_check.py 共用）。"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma"}


def ffmpeg_exe() -> str:
    p = ROOT / "tools" / "FFMPEG_PATH.txt"
    if p.exists():
        exe = p.read_text(encoding="utf-8").strip()
        if exe and Path(exe).exists():
            return exe
    return "ffmpeg"


def load_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """讀成單聲道 float32。sr=None 保留原取樣率；否則重採樣到 sr。"""
    path = Path(path)
    y = None
    file_sr = None
    try:
        import soundfile as sf
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = data.mean(axis=1)
    except Exception:
        # soundfile 讀不了（例如 m4a）就用 ffmpeg 解碼
        target = sr or 44100
        cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(target),
               "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        y = np.frombuffer(raw, dtype=np.float32).copy()
        file_sr = target
    if sr is not None and file_sr != sr:
        import soxr
        y = soxr.resample(y, file_sr, sr, quality="HQ").astype(np.float32)
        file_sr = sr
    return y.astype(np.float32), int(file_sr)


def expand_paths(items: list[str], recursive: bool = False) -> list[Path]:
    """檔案或資料夾 → 音檔清單（資料夾內依檔名排序）。"""
    out: list[Path] = []
    for it in items or []:
        p = Path(it)
        if p.is_dir():
            it_files = p.rglob("*") if recursive else p.glob("*")
            out.extend(sorted(f for f in it_files if f.is_file() and f.suffix.lower() in AUDIO_EXTS))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"找不到：{it}")
    return out


@dataclass
class AudioItem:
    name: str
    path: Path
    start: float | None = None  # 秒；None 表示整檔
    end: float | None = None

    def load(self, sr: int) -> np.ndarray:
        y, _ = load_audio(self.path, sr)
        s = 0 if self.start is None else int(round(self.start * sr))
        e = len(y) if self.end is None else int(round(self.end * sr))
        return y[s:e]

    def describe(self) -> dict:
        d = {"name": self.name, "path": str(self.path)}
        if self.start is not None or self.end is not None:
            d["start_sec"] = self.start
            d["end_sec"] = self.end
        return d


def quiet_split_point(y: np.ndarray, sr: int, search_frac: float = 0.1, win_sec: float = 0.3) -> float:
    """在中點 ±search_frac 範圍內找最安靜的 win_sec 視窗中心，回傳秒數（避免把字切一半）。"""
    n = len(y)
    mid = n // 2
    lo = max(0, int(mid - search_frac * n))
    hi = min(n, int(mid + search_frac * n))
    win = max(1, int(win_sec * sr))
    hop = max(1, int(0.01 * sr))
    best, best_e = mid, None
    for s in range(lo, max(lo + 1, hi - win), hop):
        e = float(np.mean(y[s:s + win] ** 2))
        if best_e is None or e < best_e:
            best_e, best = e, s + win // 2
    return best / sr


def halves(path: Path, label: str | None = None, split: float | None = None) -> list[AudioItem]:
    """把一個檔案切成前半、後半（切點預設在中間附近最安靜處；可指定 split 秒數）。"""
    y, sr = load_audio(path, 16000)
    t = quiet_split_point(y, sr) if split is None else split
    dur = len(y) / sr
    label = label or path.stem
    return [AudioItem(f"{label}[前半]", path, 0.0, round(t, 3)),
            AudioItem(f"{label}[後半]", path, round(t, 3), round(dur, 3))]
