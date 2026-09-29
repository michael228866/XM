@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
rem One USER launch runs the approved reference/control or improvement workflow plus validation.
set "XM_USER_TRAINING_BAT=RUN_TRAINING_V1"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [失敗] 找不到核准的 Python 環境
  echo TRAIN_STATUS=NOT_STARTED VALIDATOR_STATUS=NOT_RUN FINAL_STATUS=FAIL
  goto done
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0manual_training_launcher_v1.py"
:done
echo 請按任意鍵關閉
pause >nul
endlocal
