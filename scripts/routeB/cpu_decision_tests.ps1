# GPU 還不能用時的 CPU 小測試：決定哪些指令／模式值得正式跑。
# 用法：powershell -ExecutionPolicy Bypass -File routeB\scripts\cpu_decision_tests.ps1
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$env:PYTHONIOENCODING = 'utf-8'
$CV = '.\envs\cosyvoice\Scripts\python.exe'
$OV = '.\envs\omnivoice\Scripts\python.exe'
$log = 'routeB\logs\cpu_decision.log'
"start $(Get-Date)" | Out-File $log -Encoding utf8
# 1) CosyVoice3：三種指令 × 去噪 22 秒 prompt × 第 4、7 句（有逗號、較長）
& $CV routeB\scripts\gen_cosyvoice.py --device cpu --prompt clean22=routeA\sources\ref_clean_06-08-07.wav --instructs base warm elder --ids 4 7 --seeds 1 --variant cpu_ *>> $log
# 2) CosyVoice3：第 1 句用家人原稿「大家」
& $CV routeB\scripts\gen_cosyvoice.py --device cpu --prompt clean22=routeA\sources\ref_clean_06-08-07.wav --instructs base --ids 1 --seeds 1 --text-field hanzi_orig --variant cpu_orig_ *>> $log
# 3) OmniVoice 診斷（第 4 句）：空逐字稿／華語逐字稿／不給參考音／數字調
& $OV routeB\scripts\gen_omnivoice.py --device cpu --ref a06=stage1\segments\agong_06.wav --ids 4 --seeds 1 --variant cpu_empty *>> $log
& $OV routeB\scripts\gen_omnivoice.py --device cpu --ref a06=stage1\segments\agong_06.wav --ref-text-mode zh --ids 4 --seeds 1 --variant cpu_zh *>> $log
& $OV routeB\scripts\gen_omnivoice.py --device cpu --ref noref= --ids 4 1 --seeds 1 --variant cpu_noref *>> $log
& $OV routeB\scripts\gen_omnivoice.py --device cpu --ref a06=stage1\segments\agong_06.wav --text-field tailo_num --ids 4 --seeds 1 --variant cpu_num *>> $log
"end $(Get-Date)" | Out-File $log -Append -Encoding utf8
