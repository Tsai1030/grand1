"""步驟 6：原始／只去噪／完整增強 各版跟原始錄音比聲紋，並估雜訊程度。

輸出 stage1/version_compare.json。
聲紋比較兩種：
  same_content：該版整段 vs 原始整段（同內容，只看「處理後音色有沒有被改」）
  cross_half  ：該版前半 vs 原始後半、該版後半 vs 原始前半 的平均（不同內容，跟「原始前半 vs 原始後半」同基準）
"""
import json

import numpy as np

from common import ORIG_WAV, ROOT, STAGE1, VERSIONS, noise_stats, vad_regions
from audio_io import AudioItem, load_audio, quiet_split_point
from speaker_sim import SpeakerEmbedder, cosine


def main():
    emb = SpeakerEmbedder("campplus", "auto")
    emb2 = SpeakerEmbedder("eres2netv2", "auto")
    y16, _ = load_audio(ORIG_WAV, 16000)
    dur = len(y16) / 16000
    t_mid = round(quiet_split_point(y16, 16000), 3)
    regions = vad_regions(load_audio(VERSIONS["denoise"], 16000)[0])

    def items(path):
        return (AudioItem("full", path), AudioItem("h1", path, 0.0, t_mid), AudioItem("h2", path, t_mid, round(dur, 3)))

    res = {"split_sec": t_mid, "duration_sec": round(dur, 3), "vad_regions": len(regions), "versions": {}}
    o_full, o_h1, o_h2 = items(ORIG_WAV)
    for name, path in VERSIONS.items():
        if not path.exists():
            continue
        v_full, v_h1, v_h2 = items(path)
        row = {"path": str(path.relative_to(ROOT))}
        for tag, e in (("campplus", emb), ("eres2netv2", emb2)):
            row[f"{tag}_same_content"] = round(cosine(e.embed(v_full), e.embed(o_full)), 4)
            row[f"{tag}_cross_half"] = round(float(np.mean([cosine(e.embed(v_h1), e.embed(o_h2)),
                                                             cosine(e.embed(v_h2), e.embed(o_h1))])), 4)
            row[f"{tag}_own_h1_vs_h2"] = round(cosine(e.embed(v_h1), e.embed(v_h2)), 4)
        y, sr = load_audio(path)
        row["sr"] = sr
        row.update(noise_stats(y, sr, regions))
        res["versions"][name] = row
        print(name, row)
    out = STAGE1 / "version_compare.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
