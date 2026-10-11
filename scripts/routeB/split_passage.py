"""把整段音檔在長停頓處切成逐句檔 s1_x.wav … 給 asr_check.py 用。"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

src, out_dir, n_expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
y, sr = sf.read(src, dtype="float32")
if y.ndim > 1:
    y = y.mean(axis=1)
hop = int(sr * 0.01)
frames = len(y) // hop
db = 20 * np.log10(np.sqrt(np.array([np.mean(y[i * hop:(i + 1) * hop] ** 2) for i in range(frames)])) + 1e-9)
silent = db < (np.percentile(db, 95) - 35)

runs, start = [], None
for i, s in enumerate(silent):
    if s and start is None:
        start = i
    if not s and start is not None:
        runs.append((start, i))
        start = None
inner = [r for r in runs if r[0] > 0 and r[1] < frames]

for min_len in (0.55, 0.5, 0.45, 0.4, 0.6, 0.65):
    gaps = sorted([r for r in inner if (r[1] - r[0]) * 0.01 >= min_len], key=lambda r: r[1] - r[0], reverse=True)
    if len(gaps) >= n_expected - 1:
        cuts = sorted(gaps[: n_expected - 1])
        break
else:
    sys.exit(f"找不到足夠的停頓：{len(inner)} 段")

out_dir.mkdir(parents=True, exist_ok=True)
bounds = [0] + [((a + b) // 2) * hop for a, b in cuts] + [len(y)]
for k in range(n_expected):
    seg = y[bounds[k]:bounds[k + 1]]
    sf.write(out_dir / f"s{k + 1}_part.wav", seg, sr)
    print(f"s{k + 1}: {len(seg) / sr:.2f}s")
