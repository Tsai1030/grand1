# 祝福文：漢字／台羅（待家人確認）

依據：教育部《臺灣台語常用詞辭典》（sutian.moe.edu.tw，「用臺灣台語查詞目」），2026-10-11 逐詞查詢。查詢腳本 `routeB/scripts/sutian_lookup.py`，原始頁面與結果存在 `routeB/tailo_lookup/`（`lookup.json`、`su_*.html`）。
（查詢時該網站的 TLS 憑證已過期，瀏覽器會跳警告；內容照常可讀。）

標記：**【改】**＝跟 lead 草稿不同；**【確認】**＝辭典有兩種以上唸法，或辭典沒有收這個詞、是自己組的，請家人照阿公平常的講法決定。
台羅只標本調（不標變調），輕聲用「--」。數字調由調符自動轉換（`routeB/scripts/tailo_numeric.py`）。

| # | 漢字 | 台羅（調符） | 台羅（數字調） | 華語意思 | 標記 |
| --- | --- | --- | --- | --- | --- |
| 1 | 我是阿公。 | Guá sī a-kong. | Gua2 si7 a1-kong1. | 我是阿公。 | 【改】使用者決定刪掉「大家好」（見下） |
| 2 | 阿公佇遮一切攏好，恁毋免煩惱。 | A-kong tī tsia it-tshè lóng hó, lín m̄-bián huân-ló. | A1-kong1 ti7 tsia1 it4-tshe3 long2 ho2, lin2 m7-bian2 huan5-lo2. | 阿公在這裡一切都好，你們不用擔心。 | |
| 3 | 多謝恁這幾冬來的照顧佮陪伴。 | To-siā lín tsit kuí tang lâi ê tsiàu-kòo kah puê-phuānn. | To1-sia7 lin2 tsit4 kui2 tang1 lai5 e5 tsiau3-koo3 kah4 pue5-phuann7. | 謝謝你們這幾年來的照顧和陪伴。 | |
| 4 | 阿公希望恁以後的日子，逐工攏歡歡喜喜。 | A-kong hi-bāng lín í-āu ê ji̍t-tsí, ta̍k-kang lóng huann-huann-hí-hí. | A1-kong1 hi1-bang7 lin2 i2-au7 e5 jit8-tsi2, tak8-kang1 long2 huann1-huann1-hi2-hi2. | 阿公希望你們以後的日子，每天都開開心心。 | 【確認】日子 ji̍t-tsí／li̍t-tsí |
| 5 | 身體愛顧予好，食予飽，睏予好。 | Sin-thé ài kòo hōo hó, tsia̍h hōo pá, khùn hōo hó. | Sin1-the2 ai3 koo3 hoo7 ho2, tsiah8 hoo7 pa2, khun3 hoo7 ho2. | 身體要照顧好，要吃飽，要睡好。 | |
| 6 | 一家伙仔愛相疼惜，有代誌愛鬥相共。 | Tsi̍t-ke-hué-á ài sio-thiànn-sioh, ū tāi-tsì ài tàu-sann-kāng. | Tsit8-ke1-hue2-a2 ai3 sio1-thiann3-sioh4, u7 tai7-tsi3 ai3 tau3-sann1-kang7. | 一家人要互相疼愛，有事情要互相幫忙。 | 【確認】一家伙仔 hué／hé；相疼惜是組合詞 |
| 7 | 阿公會佇天頂保庇恁，平平安安。 | A-kong ē tī thinn-tíng pó-pì lín, pîng-pîng-an-an. | A1-kong1 e7 ti7 thinn1-ting2 po2-pi3 lin2, ping5-ping5-an1-an1. | 阿公會在天上保佑你們，平平安安。 | 【確認】會 ē／uē；平平安安是疊詞 |
| 8 | 阿公足疼恁的。 | A-kong tsiok thiànn lín--ê. | A1-kong1 tsiok4 thiann3 lin2--e5. | 阿公非常疼愛你們。 | 【確認】足 tsiok／tsok |

共 8 句、93 個音節（家人原稿就是 8 句）。

## 改過的地方

| 句 | 草稿 | 改成 | 理由 |
| --- | --- | --- | --- |
| 1 | 大家好，我是阿公。 Tāi-ke hó, guá sī a-kong. | **我是阿公。 Guá sī a-kong.** | 2026-10-11 使用者決定刪掉「大家好」（`inputs/blessing_text.txt` 已更新，舊版留在 `inputs/blessing_text_v1_含大家好.txt`）。 |

查辭典時順便發現（「大家好」已刪，僅供參考）：辭典的「大家」唸 **ta-ke**，意思是「婆婆（丈夫的母親）」；「大家、眾人」要寫「逐家 ta̍k-ke」。草稿的 Tāi-ke 是照華語唸的，辭典沒收。CPU 小測試時 CosyVoice3 把「逐家好」唸到被 ASR 聽成「自家好」。之後若要加回問候語，建議寫「逐家好 ta̍k-ke hó」並請家人聽過。

## 要家人確認的地方（辭典有不同唸法，看阿公的腔）

| 句 | 詞 | 這份稿用的 | 辭典另一種 | 說明 |
| --- | --- | --- | --- | --- |
| 4 | 日子 | ji̍t-tsí | li̍t-tsí | 辭典兩種都收（j／l 是腔調差異）。 |
| 6 | 一家伙仔 | tsi̍t-ke-hué-á | tsi̍t-ke-hé-á | 辭典兩種都收（ue／e 是腔調差異）。 |
| 6 | 相疼惜 | sio-thiànn-sioh | — | 辭典沒有收「相疼惜」整個詞；是「相 sio（互相）」＋「疼惜 thiànn-sioh」組起來的。 |
| 7 | 會 | ē | uē | 辭典兩種都收。 |
| 7 | 平平安安 | pîng-pîng-an-an | — | 辭典只收「平安 pîng-an」，這裡是疊詞。注意辭典的「平平」是 pênn-pênn（＝一樣），跟這裡不同。 |
| 8 | 足 | tsiok | tsok | 辭典：「非常、很」的意思「也可唸作 tsok」。 |
| 8 | 的（句尾） | --ê（輕聲） | — | 辭典「的 --ê」：置於句尾加強語氣的助詞。數字調寫成 e5（輕聲靠「--」表示）。 |

## 其他逐詞核對結果（都跟草稿一致）

阿公 a-kong、我 guá、是 sī、好 hó、佇 tī、遮 tsia（這裡）、一切 it-tshè、攏 lóng、恁 lín、毋免 m̄-bián、煩惱 huân-ló、多謝 to-siā、這 tsit（後接量詞）、幾 kuí、冬 tang（年）、來 lâi、的 ê（所有格）、照顧 tsiàu-kòo、佮 kah、陪伴 puê-phuānn、希望 hi-bāng、以後 í-āu、逐工 ta̍k-kang、歡歡喜喜 huann-huann-hí-hí、身體 sin-thé、愛 ài、顧 kòo、予 hōo、食 tsia̍h、飽 pá、睏 khùn、有 ū、代誌 tāi-tsì、鬥相共 tàu-sann-kāng、天頂 thinn-tíng、保庇 pó-pì、疼 thiànn、疼惜 thiànn-sioh。

補充：
- 「佮」辭典白話音 kah、文讀音 kap，口語兩種都有人講；稿子用 kah。
- 「這幾冬來」：這 tsit＋幾 kuí＋冬 tang＋來 lâi，辭典沒收整個詞，是逐字組合，屬一般說法。
