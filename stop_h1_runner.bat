@echo off
rem Stops the H1 runner (windowed or hidden). Its server-side SL/TP stay active; the 48 h timeout does not.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*h1_runner_v1.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('stopped ' + $_.ProcessId) }"
pause
