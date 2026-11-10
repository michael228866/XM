@echo off
rem Demo runner for the frozen H1 C candidate, beside gemini.py. Magic 20261108, 0.01 lot. Log: h1_runner_signal_log.csv
rem Account identity comes from local_demo_account.bat (not committed).
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist local_demo_account.bat (echo Missing local_demo_account.bat ^(sets XM_EXPECTED_LOGIN / XM_EXPECTED_SERVER^) & pause & exit /b 1)
call local_demo_account.bat
"C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" -u h1_runner_v1.py
pause
