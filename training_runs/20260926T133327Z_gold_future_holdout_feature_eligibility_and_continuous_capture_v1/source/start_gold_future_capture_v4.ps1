$ErrorActionPreference = 'Stop'
$captureRoot = Join-Path $PSScriptRoot 'future_holdout\gold_s4_v4'
$healthRoot = Join-Path $captureRoot 'health'
New-Item -ItemType Directory -Force -Path $healthRoot | Out-Null
if ((Get-ChildItem -LiteralPath $captureRoot -Filter 'PAUSED*.json') -or
    (Get-ChildItem -LiteralPath $healthRoot -Filter 'PAUSED*.json')) {
    Write-Output 'PAUSED: explicit recertification required'
    exit 0
}
$stopRequest = Join-Path $healthRoot 'stop.request'
if (Test-Path -LiteralPath $stopRequest) { Remove-Item -LiteralPath $stopRequest }
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'gold_future_capture_supervisor_v1.py') --run
exit $LASTEXITCODE
