@echo off
setlocal

REM Try local venv python
if exist ".venv\Scripts\python.exe" (
    echo Using virtual environment...
    ".venv\Scripts\python.exe" main.py
    exit /b %errorlevel%
)

REM Fall back to system python
echo Virtual environment not found, using system Python...
python main.py
