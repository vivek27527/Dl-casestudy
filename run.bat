@echo off
cd /d "%~dp0"
echo Starting Automated Industrial Quality Inspection...
venv\Scripts\python.exe run_all.py %*
pause
