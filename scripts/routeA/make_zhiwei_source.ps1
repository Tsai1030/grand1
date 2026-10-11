# 暫用來源（b）：Windows 內建 zh-TW 男聲 Microsoft Zhiwei（OneCore 語音，透過 WinRT 呼叫）。
# 講的是華語，不是台語：只能拿來比音色相似度和雜音，不能判斷台語聲調。
# 用法：powershell -ExecutionPolicy Bypass -File routeA\scripts\make_zhiwei_source.ps1 -Out routeA\sources\xxx.wav
param(
    [string]$Out = (Join-Path (Split-Path -Parent $PSScriptRoot) 'sources\TEMP_source_zhiwei_zhTW_raw.wav'),
    [string]$Text = '逐家好，我是恁的阿公。今仔日看到大家都很好，阿公心內真歡喜。希望大家身體健康，平安快樂，讀書、工作都順順利利。要記得互相照顧，有空就回來家裡坐坐，陪阿嬤吃飯聊天。阿公永遠疼你們，祝福大家每一天都歡歡喜喜。'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$resultType) {
    $t = $asTaskGeneric.MakeGenericMethod($resultType).Invoke($null, @($op))
    $t.Wait() | Out-Null
    $t.Result
}

$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -like '*Zhiwei*' } | Select-Object -First 1
if (-not $voice) { throw '找不到 Microsoft Zhiwei 語音' }
$synth.Voice = $voice
$synth.Options.SpeakingRate = 0.9
$stream = Await ($synth.SynthesizeTextToStreamAsync($Text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$size = [uint32]$stream.Size
$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
$null = Await ($reader.LoadAsync($size)) ([uint32])
$bytes = New-Object byte[] $size
$reader.ReadBytes($bytes)
New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
[System.IO.File]::WriteAllBytes($Out, $bytes)
Write-Output ("voice={0} bytes={1} -> {2}" -f $voice.DisplayName, $size, $Out)
