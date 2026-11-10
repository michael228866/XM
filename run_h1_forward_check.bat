@echo off
rem Forward check of the frozen H1 candidate (frozen\h1_C_long_v1): trades since its cutoff vs always-long.
rem Reads MT5 bars only, never trades. Results: forward_reports\<timestamp>_h1_C_forward\summary.json
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" -B h1_forward_check_v1.py
pause
