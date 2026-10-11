"""台羅調符 → 數字調（教育部臺羅規則）。輕聲的「--」保留，後面音節照原標的調號給數字。
用法：python tailo_numeric.py "Ta̍k-ke hó, guá sī a-kong."
"""
import re
import sys
import unicodedata

MARK2TONE = {"́": 2, "̀": 3, "̂": 5, "̌": 6, "̄": 7, "̍": 8, "̋": 9}


def syl_to_num(s):
    d = unicodedata.normalize("NFD", s)
    tone = None
    for m, t in MARK2TONE.items():
        if m in d:
            tone = t
            d = d.replace(m, "")
    base = unicodedata.normalize("NFC", d)
    if tone is None:
        tone = 4 if re.search(r"(p|t|k|h)$", base.lower()) else 1
    return f"{base}{tone}"


def to_numeric(text):
    # 音節＝連續的拉丁字母（含組合符號）；保留 - 、-- 、標點、空白
    def rep(m):
        return syl_to_num(m.group(0))
    d = unicodedata.normalize("NFD", text)
    out = re.sub(r"[A-Za-z̀-ͯⁿ]+", rep, d)
    return unicodedata.normalize("NFC", out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for a in sys.argv[1:]:
        print(to_numeric(a))
