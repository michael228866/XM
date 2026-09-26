$ErrorActionPreference = 'Stop'
$healthRoot = Join-Path $PSScriptRoot 'future_holdout\gold_s4_v4\health'
$heartbeat = Join-Path $healthRoot 'heartbeat.json'
if (Test-Path -LiteralPath $heartbeat) { Get-Content -LiteralPath $heartbeat -Encoding UTF8 }
else { Write-Output 'NO_HEARTBEAT' }
$captureTask = Get-ScheduledTask -TaskName 'GOLD-Future-Holdout-Capture-v4' -ErrorAction SilentlyContinue
if ($captureTask) { Write-Output ('TASK_STATE=' + $captureTask.State) }
else { Write-Output 'TASK_STATE=NOT_INSTALLED' }
