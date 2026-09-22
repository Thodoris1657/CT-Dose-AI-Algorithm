@echo off
REM One-time setup: creates a virtual environment and installs dependencies
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
echo.
echo Setup complete. Start the app with run_app.bat
pause
