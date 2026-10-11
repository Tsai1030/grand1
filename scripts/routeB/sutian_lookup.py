"""逐詞查教育部《臺灣台語常用詞辭典》（sutian.moe.edu.tw，台語詞目查詢 lui=tai_su）。
只抓「完全符合」的詞目：詞目、音讀、釋義。結果存 routeB/tailo_lookup/lookup.json。

注意：2026-10-11 查詢時該站 TLS 憑證已過期（curl/瀏覽器都報 certificate has expired），
這裡只讀公開辭典頁面，所以關掉憑證驗證（不送任何資料）。
"""
import html
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "tailo_lookup"
OUT.mkdir(exist_ok=True)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

WORDS = sys.argv[1:] or [
    "大家", "好", "我", "是", "阿公",
    "佇", "遮", "一切", "攏", "恁", "毋免", "煩惱",
    "多謝", "這", "幾", "冬", "來", "的", "照顧", "佮", "陪伴",
    "希望", "以後", "日子", "逐工", "歡喜", "歡歡喜喜",
    "身體", "愛", "顧", "予", "食", "飽", "睏",
    "一家伙仔", "相", "疼惜", "相疼", "有", "代誌", "鬥相共",
    "會", "天頂", "保庇", "平安", "平平安安",
    "足", "疼", "毋", "免", "這幾冬", "伴",
]


def fetch(word):
    url = "https://sutian.moe.edu.tw/zh-hant/tshiau/?" + urllib.parse.urlencode({"lui": "tai_su", "tsha": word})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        return url, r.read().decode("utf-8", "replace")


def text_of(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def parse_exact(page):
    # 「完全符合」區塊：到「部分符合」或頁尾為止
    m = re.search(r"完全符合(.*?)(部分符合|</main>|$)", page, re.S)
    if not m:
        return []
    block = m.group(1)
    rows = []
    # 表格版：每個 <tr> 有 序號、詞目、音讀、釋義
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", block, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) >= 4:
            no, su, im, gi = (text_of(x) for x in tds[:4])
            link = re.search(r'href="(/zh-hant/su/\d+/)"', tr)
            im = re.sub(r"播放音讀\S*", "", im).strip()
            rows.append({"no": no, "su": su, "im": im, "gi": gi,
                         "url": ("https://sutian.moe.edu.tw" + link.group(1)) if link else ""})
    return rows


res = {}
for w in WORDS:
    try:
        url, page = fetch(w)
        (OUT / f"su_{w}.html").write_text(page, encoding="utf-8")
        rows = parse_exact(page)
        res[w] = {"query": url, "exact": rows}
        print(w, "|", " ; ".join(f"{r['su']} {r['im']} 〔{r['gi'][:40]}〕" for r in rows) or "（無完全符合）", flush=True)
    except Exception as e:  # noqa
        res[w] = {"error": repr(e)}
        print(w, "ERROR", e, flush=True)
    time.sleep(0.5)

old = {}
p = OUT / "lookup.json"
if p.exists():
    old = json.loads(p.read_text(encoding="utf-8"))
old.update(res)
p.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
