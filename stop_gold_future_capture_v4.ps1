$ErrorActionPreference = 'Stop'
$healthRoot = Join-Path $PSScriptRoot 'future_holdout\gold_s4_v4\health'
New-Item -ItemType Directory -Force -Path $healthRoot | Out-Null
[System.IO.File]::WriteAllText((Join-Path $healthRoot 'stop.request'), [DateTime]::UtcNow.ToString('o'))
Write-Output 'STOP_REQUESTED; current sealed-write cycle may finish; no force termination'
