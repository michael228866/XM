$ErrorActionPreference = 'Stop'
$taskName = 'GOLD-Future-Holdout-Capture-v4'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'TASK_ALREADY_EXISTS; inspect existing configuration; no overwrite'
}
$currentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$startScript = Join-Path $PSScriptRoot 'start_gold_future_capture_v4.ps1'
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$action = New-ScheduledTaskAction -Execute $powershell -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $startScript + '"') -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentIdentity
$principal = New-ScheduledTaskPrincipal -UserId $currentIdentity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Raw GOLD holdout capture only; no models, strategies or trading; policy pauses require adjudication' | Out-Null
Write-Output 'SCHEDULED_TASK_STATUS=INSTALLED'
