"""把各資料夾的 summary.json 整理成 Markdown 表（給 eval_report.md 用）。

用法：python make_tables.py [--out routeA/eval/tables.md]
聲調風險（F0 輪廓跟原音的整段相關 r）：≥ 0.70 低、0.45–0.70 中、< 0.45 高。
  校準：同一段原音只移調 −2/−4 半音 → r≈0.85；整段變速 1.1 倍 → 0.92；不同內容的阿公參考音 → 0.22。
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RA = os.path.abspath(os.path.join(HERE, ".."))

SECTIONS = [
    ("零樣本（不訓練）", "zero_shot", None),
    ("v1 微調：去噪片段 stage1/segments", "finetune/v1_denoise", None),
    ("v1 微調：原始片段 stage1/segments_orig", "finetune/v1_orig", None),
    ("v2 微調（只訓 CFM）：去噪片段", "finetune/v2_denoise", None),
    ("參考音／節奏／種子實驗", "experiments", None),
    ("金屬感：只改推論設定", "experiments_metal", None),
]


def risk(r):
    if r is None:
        return "?"
    return "低" if r >= 0.70 else ("中" if r >= 0.45 else "高")


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(rows):
    hdr = ("| 檔案 | 對原始 前／後／平均 | 對去噪 前／後／**平均** | 音高跳動 | 大聲字/10s | F0 r 整段／字調 | 聲調風險 | 語速 字/秒 | 停頓比例 |\n"
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    out = []
    for r in rows:
        out.append(
            f"| {r['file'].replace('.wav', '')} | {fmt(r['sim_raw_first'])}／{fmt(r['sim_raw_second'])}／{fmt(r['sim_raw_mean'])} "
            f"| {fmt(r['sim_dn_first'])}／{fmt(r['sim_dn_second'])}／**{fmt(r['sim_dn_mean'])}** "
            f"| {fmt(r['pitch_jump_ratio'], 4)} | {fmt(r['loud_per_10s'], 2)} "
            f"| {fmt(r['f0_r_global'], 2)}／{fmt(r['f0_r_local'], 2)} | {risk(r['f0_r_global'])} "
            f"| {fmt(r.get('speech_rate'), 2)} | {fmt(r.get('pause_ratio'), 2)} |")
    return hdr + "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RA, "eval", "tables.md"))
    a = ap.parse_args()
    md = []
    for title, sub, _ in SECTIONS:
        p = os.path.join(RA, sub, "summary.json")
        if not os.path.exists(p):
            md.append(f"### {title}\n\n（尚未產生：{sub}/summary.json）\n")
            continue
        rows = json.load(open(p, encoding="utf-8"))["rows"]
        md.append(f"### {title}（`routeA/{sub}/`）\n\n" + table(rows))
    open(a.out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
