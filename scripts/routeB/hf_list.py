"""列出 HF repo 檔案與大小（只讀，不下載）。用法：python hf_list.py repo_id [repo_type]"""
import sys
from huggingface_hub import HfApi

api = HfApi()
rtype = sys.argv[2] if len(sys.argv) > 2 else "model"
for rid in sys.argv[1:2]:
    info = api.repo_info(rid, repo_type=rtype, files_metadata=True)
    print("==", rid, info.sha)
    tot = 0
    for s in info.siblings:
        sz = s.size or 0
        tot += sz
        print(f"  {s.rfilename:70s} {sz/1e6:10.1f} MB")
    print("  total", round(tot / 1e9, 2), "GB")
