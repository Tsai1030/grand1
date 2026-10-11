# 零樣本基準（不訓練）：台語原音 + 阿公 14 秒參考音
# 用法：powershell -ExecutionPolicy Bypass -File routeA\scripts\run_zero_shot.ps1 [-Source <wav>]
param(
    [string]$Source = '',
    [string]$Ref = '',
    [string]$OutDir = ''
)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Source) { $Source = Join-Path $root 'inputs\taiwanese_source_joined.wav' }
if (-not $Ref) { $Ref = Join-Path $root 'inputs\voice_reference_15s.wav' }
if (-not $OutDir) { $OutDir = Join-Path $root 'routeA\zero_shot' }
$py = Join-Path $root 'envs\seedvc\Scripts\python.exe'
$run = Join-Path $root 'routeA\scripts\run_infer.py'
$ff = Split-Path -Parent (Get-Content (Join-Path $root 'tools\FFMPEG_PATH.txt') -TotalCount 1).Trim()
$env:PATH = "$ff;$env:PATH"                     # v2 用 pydub 轉 mp3 串流，需要 ffmpeg
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force $OutDir, (Join-Path $root 'routeA\logs') | Out-Null

$jobs = @(
    @{ name = 'zs_v1_ds50_cfg0.7'; model = 'v1'; a = @('--diffusion-steps', '50', '--inference-cfg-rate', '0.7', '--f0-condition', 'False', '--auto-f0-adjust', 'False', '--fp16', 'True') },
    @{ name = 'zs_v1_ds30_cfg0.7'; model = 'v1'; a = @('--diffusion-steps', '30', '--inference-cfg-rate', '0.7', '--f0-condition', 'False', '--auto-f0-adjust', 'False', '--fp16', 'True') },
    @{ name = 'zs_v2_style-off'; model = 'v2'; a = @('--diffusion-steps', '30', '--intelligibility-cfg-rate', '0.7', '--similarity-cfg-rate', '0.7', '--convert-style', 'false', '--top-p', '0.9', '--temperature', '1.0', '--repetition-penalty', '1.0') },
    @{ name = 'zs_v2_style-on'; model = 'v2'; a = @('--diffusion-steps', '30', '--intelligibility-cfg-rate', '0.7', '--similarity-cfg-rate', '0.7', '--convert-style', 'true', '--top-p', '0.9', '--temperature', '1.0', '--repetition-penalty', '1.0') }
)
foreach ($j in $jobs) {
    $out = Join-Path $OutDir ($j.name + '.wav')
    $log = Join-Path $root ("routeA\logs\" + $j.name + '.log')
    $argv = @($run, '--model', $j.model, '--out', $out, '--seed', '1234', '--', '--source', $Source, '--target', $Ref) + $j.a
    & $py @argv *> $log
    Write-Output ("{0}: exit {1}" -f $j.name, $LASTEXITCODE)
    Get-Content $log | Select-String 'RUN_INFER_RESULT|Error|error' | Select-Object -Last 2 | ForEach-Object { $_.Line }
}
