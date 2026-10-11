# 微調（計畫書階段三第 1 部分）。每 100 步存一次檔，全部保留。
# 用法：
#   powershell -ExecutionPolicy Bypass -File routeA\scripts\run_train.ps1 -Model v1 -Data stage1\segments      -Name v1_denoise
#   powershell -ExecutionPolicy Bypass -File routeA\scripts\run_train.ps1 -Model v2 -Data stage1\segments      -Name v2_denoise
#   powershell -ExecutionPolicy Bypass -File routeA\scripts\run_train.ps1 -Model v1 -Data stage1\segments_orig -Name v1_orig
param(
    [Parameter(Mandatory = $true)][ValidateSet('v1', 'v2')][string]$Model,
    [Parameter(Mandatory = $true)][string]$Data,
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$Steps = 500,
    [int]$SaveEvery = 100,
    [int]$BatchSize = 2
)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $root 'envs\seedvc\Scripts\python.exe'
$repo = Join-Path $root 'repos\seed-vc'
$runDir = Join-Path $root ("routeA\runs\" + $Name)
$dataDir = if ([IO.Path]::IsPathRooted($Data)) { $Data } else { Join-Path $root $Data }
$env:PYTHONIOENCODING = 'utf-8'
$common = @('--dataset-dir', $dataDir, '--batch-size', "$BatchSize", '--max-steps', "$Steps", '--max-epochs', '1000',
    '--save-every', "$SaveEvery", '--num-workers', '0')
if ($Model -eq 'v1') {
    $extra = @('--config', (Join-Path $repo 'configs\presets\config_dit_mel_seed_uvit_whisper_small_wavenet.yml'))
} else {
    $extra = @('--config', (Join-Path $repo 'configs\v2\vc_wrapper.yaml'), '--train-cfm')
}
New-Item -ItemType Directory -Force $runDir | Out-Null
$log = Join-Path $runDir 'train.log'
$argv = @((Join-Path $root 'routeA\scripts\train_keepall.py'), '--model', $Model, '--run-dir', $runDir, '--') + $common + $extra
& $py -u @argv *> $log
Write-Output ("train {0} exit {1}" -f $Name, $LASTEXITCODE)
Get-Content $log | Select-String 'TRAIN_STATS|Traceback|Error' | Select-Object -Last 3 | ForEach-Object { $_.Line }
