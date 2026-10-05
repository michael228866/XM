@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
rem GOLD_S4_TRADE_ECONOMICS_V1: USER only; no model training.
set "XM_USER_TRAINING_BAT=RUN_TRAINING_V1"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [??] ?????? Python ??
  goto done
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0gold_s4_trade_economics_v1_launcher.py"
:done
echo ???????
pause >nul
endlocal
