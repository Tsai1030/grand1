"""整段音檔的評估（跟 eval_candidates.py 的整段指標同一套），用來比 TTS 整段、Seed-VC 轉換後、網頁 TTS 原音、路線 A。

指標：sim_dn／sim_raw（CAM++，對阿公去噪版／原始錄音前後半的平均）、rhythm.py 的語速與停頓、
F0 起伏（3 秒片段中位數）、跳動比例、每 10 秒大聲字、text_rate（字數 ÷ 第一個到最後一個說話幀的時間）。
用法：python passage_eval.py --files 名稱=路徑=字數 ... --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_candidates as ec  # noqa: E402
from eval_candidates import AGONG_DENOISE_WAV, AGONG_RAW_WAV, SpeakerEmbedder, cosine, halves  # noqa: E402


def speech_span(y16, sr=16000):
    hop = 160
    rms = librosa.feature.rms(y=y16, frame_length=400, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    thr = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 6)
    idx = np.where(db > thr)[0]
    return (idx[-1] - idx[0] + 1) * hop / sr if len(idx) else len(y16) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    emb = SpeakerEmbedder("campplus", "cpu")
    raw_h = halves(AGONG_RAW_WAV, "阿公原始")
    dn_h = halves(AGONG_DENOISE_WAV, "阿公去噪", split=raw_h[0].end)
    ref_raw = [emb.embed(h) for h in raw_h]
    ref_dn = [emb.embed(h) for h in dn_h]

    def sims(y16):
        e = emb.embed_array(y16)
        return (round(float(np.mean([cosine(e, r) for r in ref_dn])), 4),
                round(float(np.mean([cosine(e, r) for r in ref_raw])), 4))

    rhythm = ec.load_rhythm()
    out = {}
    for spec in args.files:
        name, path, nsyl = spec.split("=")
        p = (ec.ROOT / path) if not Path(path).is_absolute() else Path(path)
        m = ec.passage_metrics(p, sims, rhythm)
        y16, _ = librosa.load(str(p), sr=16000)
        m["text_rate"] = round(int(nsyl) / speech_span(y16), 2)
        out[name] = m
        print(name, json.dumps(m, ensure_ascii=False), flush=True)
    Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
