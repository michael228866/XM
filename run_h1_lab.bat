@echo off
rem Train and compare the 4 pre-registered H1 strategies in h1_lab_v1.py. Reads MT5 bars only, never trades.
rem Each run writes a new timestamped folder under forward_reports; results are in report.json there.
rem Extra arguments pass through, e.g.  run_h1_lab.bat --cross   (adds USDX/silver/S&P 500/VIX features)
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%i
set OUT=forward_reports\%STAMP%_h1_lab_v1

echo Output: %OUT%
"%PY%" -B h1_lab_v1.py --output "%OUT%" %*
if errorlevel 1 (
    echo.
    echo FAILED with exit code %errorlevel%
) else (
    echo.
    echo Done. Results: %OUT%\report.json
)
pause
