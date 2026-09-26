$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'stop_gold_future_capture_v4.ps1')
Unregister-ScheduledTask -TaskName 'GOLD-Future-Holdout-Capture-v4' -Confirm:$false -ErrorAction Stop
Write-Output 'SCHEDULED_TASK_STATUS=NOT_INSTALLED; evidence retained'
