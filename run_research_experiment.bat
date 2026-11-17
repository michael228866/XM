@echo off
rem Runs one preregistered research experiment from research_memory\ledger.jsonl (user-started).
rem Usage: run_research_experiment.bat E-001   (results: training_runs\<run id>, RESEARCH_MEMORY.md)
rem Interpreter: the repository .venv (Python 3.11, MANUAL_TRAINING_POLICY_V1.md); no fallback to another Python.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set "PY=%~dp0.venv\Scripts\python.exe"
if "%~1"=="" (
  echo Usage: run_research_experiment.bat E-001
  pause
  exit /b 1
)
if not exist "%PY%" (
  echo Required interpreter not found: %PY%
  echo Create the repository .venv ^(Python 3.11^); this script does not fall back to another Python.
  pause
  exit /b 1
)
"%PY%" -B research_run_v1.py %1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo research_run_v1.py failed with exit code %RC%
pause
exit /b %RC%
