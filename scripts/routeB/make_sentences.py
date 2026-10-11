"""祝福文逐句資料（唯一來源）→ routeB/sentences.json（給生成與評估用）。
台羅依教育部《臺灣台語常用詞辭典》逐詞核對，查詢結果在 routeB/tailo_lookup/lookup.json。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tailo_numeric import to_numeric  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

S = [
    # id, 漢字（家人原稿）, 漢字（辭典用字，給 TTS 用）, 台羅調符, 華語意思
    # 第 1 句：使用者 2026-10-11 決定刪掉「大家好」（inputs/blessing_text.txt 已更新；舊版 blessing_text_v1_含大家好.txt）
    (1, "我是阿公。", "我是阿公。",
     "Guá sī a-kong.", "我是阿公。"),
    (2, "阿公佇遮一切攏好，恁毋免煩惱。", "阿公佇遮一切攏好，恁毋免煩惱。",
     "A-kong tī tsia it-tshè lóng hó, lín m̄-bián huân-ló.", "阿公在這裡一切都好，你們不用擔心。"),
    (3, "多謝恁這幾冬來的照顧佮陪伴。", "多謝恁這幾冬來的照顧佮陪伴。",
     "To-siā lín tsit kuí tang lâi ê tsiàu-kòo kah puê-phuānn.", "謝謝你們這幾年來的照顧和陪伴。"),
    (4, "阿公希望恁以後的日子，逐工攏歡歡喜喜。", "阿公希望恁以後的日子，逐工攏歡歡喜喜。",
     "A-kong hi-bāng lín í-āu ê ji̍t-tsí, ta̍k-kang lóng huann-huann-hí-hí.", "阿公希望你們以後的日子，每天都開開心心。"),
    (5, "身體愛顧予好，食予飽，睏予好。", "身體愛顧予好，食予飽，睏予好。",
     "Sin-thé ài kòo hōo hó, tsia̍h hōo pá, khùn hōo hó.", "身體要照顧好，要吃飽，要睡好。"),
    (6, "一家伙仔愛相疼惜，有代誌愛鬥相共。", "一家伙仔愛相疼惜，有代誌愛鬥相共。",
     "Tsi̍t-ke-hué-á ài sio-thiànn-sioh, ū tāi-tsì ài tàu-sann-kāng.", "一家人要互相疼愛，有事情要互相幫忙。"),
    (7, "阿公會佇天頂保庇恁，平平安安。", "阿公會佇天頂保庇恁，平平安安。",
     "A-kong ē tī thinn-tíng pó-pì lín, pîng-pîng-an-an.", "阿公會在天上保佑你們，平平安安。"),
    (8, "阿公足疼恁的。", "阿公足疼恁的。",
     "A-kong tsiok thiànn lín--ê.", "阿公非常疼愛你們。"),
]


def n_syl(tailo):
    import re
    return len([x for x in re.split(r"[\s\-,.!?]+", tailo) if x])


rows = []
for i, orig, hj, tl, zh in S:
    rows.append({
        "id": i, "hanzi_orig": orig, "hanzi": hj, "tailo": tl, "tailo_num": to_numeric(tl),
        "zh": zh, "syllables": n_syl(tl),
    })
out = ROOT / "routeB" / "sentences.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
for r in rows:
    print(r["id"], r["syllables"], r["hanzi"], "|", r["tailo_num"])
print("total syllables", sum(r["syllables"] for r in rows))
