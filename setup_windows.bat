@echo off
REM One-time setup: virtual environment, dependencies, and the first training run.
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
echo.
echo Setup complete. Start the application with run_app.bat
pause
