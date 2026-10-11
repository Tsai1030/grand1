"""台羅本調 -> 變調後實際調值（標調符）。分組用 lead 給的 | 版本。輸出 routeB/sentences_sandhi.json（多一個 tailo_sandhi 欄位）。"""
import json, re, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
1: "Guá sī a-kong.",
2: "A-kong | tī tsia it-tshè lóng hó, | lín m̄-bián huân-ló.",
3: "To-siā lín | tsit kuí tang lâi ê tsiàu-kòo | kah puê-phuānn.",
4: "A-kong | hi-bāng lín í-āu ê ji̍t-tsí, | ta̍k-kang | lóng huann-huann-hí-hí.",
5: "Sin-thé | ài kòo hōo hó, | tsia̍h hōo pá, | khùn hōo hó.",
6: "Tsi̍t-ke-hué-á | ài sio-thiànn-sioh, | ū tāi-tsì | ài tàu-sann-kāng.",
7: "A-kong | ē tī thinn-tíng | pó-pì lín, | pîng-pîng-an-an.",
8: "A-kong | tsiok thiànn lín--ê.",
}
MARK = {"́": 2, "̀": 3, "̂": 5, "̄": 7, "̍": 8}
COMB = {2: "́", 3: "̀", 5: "̂", 7: "̄", 8: "̍"}


def parse(s):
    d = unicodedata.normalize("NFD", s)
    tone = None
    base = ""
    for ch in d:
        if ch in MARK: tone = MARK[ch]
        else: base += ch
    if tone is None:
        tone = 4 if base.lower()[-1] in "ptkh" else 1
    return base, tone


def sandhi(base, tone):
    stop = base.lower()[-1] in "ptk"
    if tone == 4: return 8 if stop else 2
    if tone == 8: return 4 if stop else 3
    return {1: 7, 7: 3, 3: 2, 2: 1, 5: 7}[tone]


def render(base, tone):
    if tone in (1, 4): return unicodedata.normalize("NFC", base)
    l = base.lower()
    if "a" in l: i = l.index("a")
    elif "oo" in l: i = l.index("o")
    elif "o" in l: i = l.index("o")
    elif "e" in l: i = l.index("e")
    else:
        iu = [k for k, c in enumerate(l) if c in "iu"]
        if iu: i = iu[-1]
        else: i = 0  # m / ng
    return unicodedata.normalize("NFC", base[:i + 1] + COMB[tone] + base[i + 1:])


def convert(group_text):
    out = []
    for grp in [g.strip() for g in group_text.split("|")]:
        toks = [(m.group(0), m.start()) for m in re.finditer(r"[^\s\-.,]+", grp)]
        neutral = []
        for k, (t, st) in enumerate(toks):
            neutral.append(k > 0 and grp[toks[k - 1][1] + len(toks[k - 1][0]):st].startswith("--"))
        last = max(k for k in range(len(toks)) if not neutral[k])
        res, pos = "", 0
        for k, (t, st) in enumerate(toks):
            res += grp[pos:st]
            b, tn = parse(t)
            res += render(b, sandhi(b, tn)) if (k < last and not neutral[k]) else unicodedata.normalize("NFC", t)
            pos = st + len(t)
        res += grp[pos:]
        out.append(res)
    return " ".join(out)


if __name__ == "__main__":
    s = json.loads((ROOT / "routeB/sentences.json").read_text(encoding="utf-8"))
    for x in s:
        x["tailo_sandhi"] = convert(GROUPS[x["id"]])
        print(x["id"], x["tailo_sandhi"])
    (ROOT / "routeB/sentences_sandhi.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
