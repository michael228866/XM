@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
rem GOLD_S4_ENTRY_EDGE_DECOMPOSITION_V1: USER only; no training or entry filtering.
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [FAIL] Repository Python is missing.
  goto done
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0gold_s4_entry_edge_decomposition_v1_launcher.py" --status
:done
echo Press any key to close.
pause >nul
endlocal
