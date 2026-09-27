$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$stopRequest = Join-Path $PSScriptRoot 'future_holdout\gold_s4_v4\health\stop.request'
if (Test-Path -LiteralPath $stopRequest) { Remove-Item -LiteralPath $stopRequest }
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'gold_future_capture_supervisor_v1_1.py') --run
exit $LASTEXITCODE
