"""步驟 13：stage1/manifest.tsv（一段一行）。台語漢字、台羅留空給家人填。"""
import csv
import json

from common import STAGE1


def main():
    seg = json.loads((STAGE1 / "segments.json").read_text(encoding="utf-8"))
    asr_p = STAGE1 / "asr.json"
    asr = json.loads(asr_p.read_text(encoding="utf-8"))["text"] if asr_p.exists() else {}
    rows = []
    for s in seg["segments"]:
        if s["rejected"]:
            continue
        fn = f"{s['id']}.wav"
        rows.append([f"segments/{fn}", "阿公", f"{s['duration_sec']:.2f}", f"{s['sim_to_others_main']:.4f}",
                     asr.get(fn, ""), "", ""])
    with open(STAGE1 / "manifest.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["檔名", "語者", "秒數", "聲紋相似度", "華語意思(ASR)", "台語漢字", "台羅"])
        w.writerows(rows)
    print(f"{len(rows)} rows, total {sum(float(r[2]) for r in rows):.2f}s")


if __name__ == "__main__":
    main()
