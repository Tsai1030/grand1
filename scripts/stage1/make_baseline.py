"""步驟 10：產生 eval/baseline.json（聲紋基準＋阿公本人的 prosody 參考值）。只用 CPU。"""
import json
from datetime import datetime

import numpy as np

from common import ORIG_WAV, ROOT, STAGE1, VERSIONS, locate_clip
from audio_io import AudioItem, load_audio, quiet_split_point
from speaker_sim import SpeakerEmbedder, cosine
import prosody_check as pc

REF15 = ROOT / "inputs" / "voice_reference_15s.wav"
PREV_SAME_PERSON = 0.92
PREV_KNNVC = (0.60, 0.67)


def main():
    seginfo = json.loads((STAGE1 / "segments.json").read_text(encoding="utf-8"))
    vc = json.loads((STAGE1 / "version_compare.json").read_text(encoding="utf-8"))
    kept = [s for s in seginfo["segments"] if not s["rejected"]]

    y16, _ = load_audio(ORIG_WAV, 16000)
    dur = len(y16) / 16000
    t_mid = round(quiet_split_point(y16, 16000), 3)

    # voice_reference_15s 在原始錄音中的位置（分段互相關；參考音是 37.05 秒起、停頓略剪短的連續片段）
    yr, _ = load_audio(REF15, 16000)
    off, corr = locate_clip(y16, yr[:32000], 16000)          # 開頭 2 秒定位起點
    tail = yr[-32000:]
    off_t, corr_t = locate_clip(y16, tail, 16000)            # 結尾 2 秒定位終點
    ref_start, ref_end = round(off, 3), round(off_t + len(tail) / 16000, 3)

    speaker = {}
    for model in ("campplus", "eres2netv2"):
        e = SpeakerEmbedder(model, "cpu")
        h1, h2 = AudioItem("h1", ORIG_WAV, 0.0, t_mid), AudioItem("h2", ORIG_WAV, t_mid, round(dur, 3))
        E1, E2 = e.embed(h1), e.embed(h2)
        rest = np.concatenate([y16[: int(ref_start * 16000)], y16[int(ref_end * 16000):]])
        Eref, Erest = e.embed(AudioItem("ref", REF15)), e.embed_array(rest)

        def seg_cross(dirname):
            vals = []
            for s in kept:
                es = e.embed(AudioItem(s["id"], STAGE1 / dirname / f"{s['id']}.wav"))
                mid = (s["start_sec"] + s["end_sec"]) / 2
                vals.append(cosine(es, E2 if mid < t_mid else E1))  # 跟「不含該段」的那一半比
            return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4), "n": len(vals)}

        dh1 = AudioItem("dh1", VERSIONS["denoise"], 0.0, t_mid)
        dh2 = AudioItem("dh2", VERSIONS["denoise"], t_mid, round(dur, 3))
        speaker[model] = {
            "model_id": e.conf["model_id"],
            "agong_original_first_vs_second_half": round(cosine(E1, E2), 4),
            "voice_reference_15s_vs_rest_of_original": round(cosine(Eref, Erest), 4),
            "voice_reference_15s_vs_original_first_half": round(cosine(Eref, E1), 4),
            "voice_reference_15s_vs_original_second_half": round(cosine(Eref, E2), 4),
            "denoise_first_vs_second_half": round(cosine(e.embed(dh1), e.embed(dh2)), 4),
            "segments_main_vs_other_original_half": seg_cross("segments"),
            "segments_orig_vs_other_original_half": seg_cross("segments_orig"),
            "versions_vs_original": {
                k: {"same_content": v[f"{model}_same_content"], "cross_half": v[f"{model}_cross_half"]}
                for k, v in vc["versions"].items()},
        }

    cam = speaker["campplus"]
    nb_path = STAGE1 / "noise_back_test.json"
    if nb_path.exists():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cam["denoise_plus_original_noise_cross_half"] = nb["denoise_plus_noise"]["+0dB"]["cross_half"]
    ours = cam["agong_original_first_vs_second_half"]
    ratio = ours / PREV_SAME_PERSON
    scaled = round(PREV_KNNVC[1] * ratio, 3)

    # prosody：阿公本人參考值（主力版保留片段合計），另附原始版片段與整段原始錄音
    def prosody_of(files):
        res = {str(f.relative_to(ROOT)): pc.analyze(load_audio(f, pc.SR)[0]) for f in files}
        return res, pc.pooled(res)

    main_files = [STAGE1 / "segments" / f"{s['id']}.wav" for s in kept]
    orig_files = [STAGE1 / "segments_orig" / f"{s['id']}.wav" for s in kept]
    pm, pm_pool = prosody_of(main_files)
    _, po_pool = prosody_of(orig_files)
    _, pr_pool = prosody_of([REF15])

    baseline = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": {"agong_original": str(ORIG_WAV.relative_to(ROOT)),
                   "from": "inputs/male_voice_only.mp3（ffmpeg 轉單聲道 wav，保留 44.1 kHz）",
                   "duration_sec": round(dur, 3),
                   "half_split_sec": t_mid,
                   "half_split_rule": "中點 ±10% 內最安靜的 0.3 秒視窗（與 speaker_sim.py --agong-halves 相同）",
                   "voice_reference_15s_location_in_original_sec": [ref_start, ref_end],
                   "voice_reference_15s_locate_corr": [round(corr, 3), round(corr_t, 3)],
                   "main_version": seginfo["main_version"],
                   "segments_kept": len(kept)},
        "speaker": speaker,
        "speaker_note": {
            "primary_model": "campplus",
            "our_same_person_baseline": ours,
            "previous_same_person_baseline": PREV_SAME_PERSON,
            "ratio_ours_to_previous": round(ratio, 4),
            "previous_knnvc": list(PREV_KNNVC),
            "knnvc_threshold_scaled": scaled,
            "summary_zh": (
                f"CAM++ 量阿公原始錄音前半 vs 後半＝{ours:.3f}，跟之前雲端的 0.92 很接近（比值 {ratio:.3f}），"
                f"0.67 門檻可直接沿用；若要按比例換算，約為 {scaled:.3f}。"
                f"注意：乾淨的阿公聲音（只去噪版）跟『有雜音的原始錄音』另一半比只有 "
                f"{cam['versions_vs_original']['denoise']['cross_half']:.3f}"
                + (f"（把原始背景雜音加回去就回到 {cam['denoise_plus_original_noise_cross_half']:.3f}，"
                   "表示分數下降主要是雜音被拿掉，不是音色被改）"
                   if "denoise_plus_original_noise_cross_half" in cam else "")
                + "，"
                f"主力片段（去噪、5–11 秒）跟原始錄音另一半比平均 {cam['segments_main_vs_other_original_half']['mean']:.3f}；"
                "所以乾淨的轉換結果對原始錄音前後半的分數，實際上限大概在 0.8 左右，"
                "看到 0.75–0.8 已經接近『真阿公、乾淨版』的水準。建議同時對去噪版前後半比一次"
                "（--halves stage1/enhanced/male_voice_only_denoise.wav），去噪版自己前後半互比＝"
                f"{cam['denoise_first_vs_second_half']:.3f}。"),
        },
        "prosody": {
            "agong_reference": {
                "description": "主力版（只去噪）保留片段合計；判讀：測試音不高於此水準",
                "params": pc.PARAMS | {"jump_st": 3.0, "loud_db": 8.0},
                "pooled": pm_pool,
                "per_file": pm,
            },
            "agong_segments_orig_pooled": po_pool,
            "voice_reference_15s": pr_pool,
        },
    }
    out = ROOT / "eval" / "baseline.json"
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in baseline.items() if k != "prosody"}, ensure_ascii=False, indent=1))
    print("prosody pooled main:", pm_pool)
    print("prosody pooled orig:", po_pool)
    print("prosody ref15:", pr_pool)


if __name__ == "__main__":
    main()
