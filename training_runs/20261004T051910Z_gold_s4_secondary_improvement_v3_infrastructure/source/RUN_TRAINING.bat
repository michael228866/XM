@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
rem GOLD_S4_SECONDARY_IMPROVEMENT_V3: USER double-click only; Train + Val in one workflow.
set "XM_USER_TRAINING_BAT=RUN_TRAINING_V1"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [失敗] 找不到核准的 Python 環境
  echo TRAIN_STATUS=NOT_STARTED VALIDATOR_STATUS=NOT_RUN FINAL_STATUS=FAIL
  goto done
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0gold_s4_secondary_improvement_v3_launcher.py"
:done
echo 請按任意鍵關閉
pause >nul
endlocal
