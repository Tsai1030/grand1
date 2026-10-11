# 正式逐句生成（GPU）。每句 3 個 seed。方法：
#   CosyVoice3 instruct2：base×expr、warm×expr、warm×clean22、warm×expr_orig（原始未去噪對照）
#   OmniVoice：voice design「male, elderly, moderate pitch」＋台羅（不給阿公參考音；見報告：空逐字稿的複製會唸壞）
# 用法：powershell -ExecutionPolicy Bypass -File routeB\scripts\run_gpu_generation.ps1
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$CV = '.\envs\cosyvoice\Scripts\python.exe'
$OV = '.\envs\omnivoice\Scripts\python.exe'
& $CV routeB\scripts\gen_cosyvoice.py --device cuda --instructs base warm --prompt expr=routeB\prompts\expr_05-02-12.wav --seeds 1 2 3 *> routeB\logs\gen_cv_A.log
& $CV routeB\scripts\gen_cosyvoice.py --device cuda --instructs warm --prompt clean22=routeA\sources\ref_clean_06-08-07.wav --prompt expr_orig=routeB\prompts\expr_05-02-12_orig.wav --seeds 1 2 3 *> routeB\logs\gen_cv_B.log
& $OV routeB\scripts\gen_omnivoice.py --device cuda --ref design= --instruct "male, elderly, moderate pitch" --rate 3.2 --seeds 1 2 3 --variant design *> routeB\logs\gen_ov_design.log
"done $(Get-Date)" | Out-File routeB\logs\gen_gpu_done.txt -Encoding utf8
