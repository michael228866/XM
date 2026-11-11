@echo off
rem Runs one preregistered research experiment from research_memory\ledger.jsonl (user-started).
rem Usage: run_research_experiment.bat E-001   (results: training_runs\<run id>, RESEARCH_MEMORY.md)
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
  echo Usage: run_research_experiment.bat E-001
  pause
  exit /b 1
)
"C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" -B research_run_v1.py %1
pause
