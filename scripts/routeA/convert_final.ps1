# 路線 A：用評估後最好的 1–3 個 Seed-VC 存檔點，重新轉換台語原音，並自動跑評估。
#
# 一般用法（在 PowerShell，任何目錄皆可）：
#   powershell -ExecutionPolicy Bypass -File C:\Users\USER\Downloads\test\routeA\convert_final.ps1
#
# 選項：
#   -Source <wav>   台語原音。預設依序找 inputs\taiwanese_source_joined.wav、根目錄的 taiwanese_source_joined.wav
#   -Ref <wav>      阿公參考音。預設用 best_checkpoints.json 裡每個版本自己的設定（沒寫就用 inputs\voice_reference_15s.wav）
#   -OutDir <dir>   輸出資料夾。預設 routeA\final\<日期_時間>
#   -Jobs <json>    要用哪些存檔點。預設 routeA\best_checkpoints.json
#   -NoEval         只轉換、不評估
#
# 輸出：<OutDir>\*.wav（每個版本一個檔）、<OutDir>\summary.tsv（相似度、F0 相關、語速等）、<OutDir>\eval\*.json
param(
    [string]$Source = '',
    [string]$Ref = '',
    [string]$OutDir = '',
    [string]$Jobs = '',
    [switch]$NoEval
)
$ErrorActionPreference = 'Stop'
$routeA = $PSScriptRoot
$root = Split-Path -Parent $routeA
$py = Join-Path $root 'envs\seedvc\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "找不到 $py，請先執行 routeA\setup\setup_env.ps1" }

if (-not $Source) {
    foreach ($c in @((Join-Path $root 'inputs\taiwanese_source_joined.wav'), (Join-Path $root 'taiwanese_source_joined.wav'))) {
        if (Test-Path $c) { $Source = $c; break }
    }
}
if (-not $Source -or -not (Test-Path $Source)) { throw '找不到台語原音：請把 taiwanese_source_joined.wav 放進 inputs\ 或專案根目錄，或用 -Source 指定。' }
$Source = (Resolve-Path $Source).Path
if (-not $Jobs) { $Jobs = Join-Path $routeA 'best_checkpoints.json' }
if (-not $OutDir) { $OutDir = Join-Path $routeA ('final\' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$env:PYTHONIOENCODING = 'utf-8'
$argv = @((Join-Path $routeA 'scripts\convert_and_eval.py'), '--jobs', $Jobs, '--source', $Source, '--out-dir', $OutDir)
if ($Ref) { $argv += @('--ref', (Resolve-Path $Ref).Path, '--ref-override') }
if ($NoEval) { $argv += '--no-eval' }

Write-Output "來源：$Source"
Write-Output "版本清單：$Jobs"
Write-Output "輸出：$OutDir"
& $py @argv
if ($LASTEXITCODE -ne 0) { throw "轉換或評估失敗（exit $LASTEXITCODE），請看 $OutDir\logs\" }
Write-Output ''
Write-Output "完成。音檔在 $OutDir；數字在 $OutDir\summary.tsv。"
Write-Output '判讀：sim_dn_mean（對阿公去噪版前後半的平均）越高越像；同一人基準 0.901。f0_r_global 越高聲調越貼近原音（v1 約 0.77）。'
