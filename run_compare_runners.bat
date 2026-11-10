@echo off
rem Closed demo trades of gemini (M1) vs H1 C since the H1 runner started, in R. Reads MT5 history only.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" -B compare_runners_v1.py %*
pause
