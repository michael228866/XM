@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "XM_USER_TRAINING_BAT=RUN_TRAINING_V1"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [失敗] 找不到核准的 Python 環境
  goto done
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0manual_training_launcher_v1.py"
:done
echo 請按任意鍵關閉
pause >nul
endlocal
