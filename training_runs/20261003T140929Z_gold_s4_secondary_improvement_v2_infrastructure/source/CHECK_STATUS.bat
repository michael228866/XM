@echo off
rem Metadata-only GOLD S4 Improvement v2 status; never opens locked holdout.
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [失敗] 找不到核准的 Python 環境
  goto done
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0gold_s4_secondary_improvement_v2_launcher.py" --status
:done
echo 請按任意鍵關閉
pause >nul
endlocal
