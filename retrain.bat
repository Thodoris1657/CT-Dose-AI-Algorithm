@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python train.py --repeats 10
pause
