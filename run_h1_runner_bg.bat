@echo off
rem Same as run_h1_runner.bat but no pause, console output appended to h1_runner_console.log. Started hidden by start_h1_runner_hidden.vbs.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist local_demo_account.bat (echo Missing local_demo_account.bat>> h1_runner_console.log & exit /b 1)
call local_demo_account.bat
"C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" -u h1_runner_v1.py >> h1_runner_console.log 2>&1
