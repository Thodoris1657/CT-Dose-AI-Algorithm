@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate ( call .venv\Scripts\activate )
echo Starting CT Dose AI - your browser will open at http://127.0.0.1:8000
python app.py
pause
