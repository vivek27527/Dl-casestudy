@echo off
cd /d "%~dp0"
echo Generating 10 Category Datatype and Severity Tables...
venv\Scripts\python.exe run_all.py --reports-only
pause
