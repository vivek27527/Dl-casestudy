@echo off
cd /d "%~dp0"
echo Launching Quality Inspection Streamlit Web App...
venv\Scripts\streamlit.exe run app.py
pause
