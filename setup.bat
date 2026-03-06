@echo off
REM Setup script for Windows

echo.
echo ======================================================================
echo SEMANTIC SEARCH SYSTEM - Windows Setup
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11+ is required but not found!
    echo Please install Python from https://www.python.org/
    exit /b 1
)

echo Step 1: Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo Created venv/
)

echo.
echo Step 2: Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo.
echo Step 3: Preparing dataset and building indices...
python src\download_dataset.py

echo.
echo ======================================================================
echo ✅ Setup Complete!
echo ======================================================================
echo.
echo To start the API server:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Run: python -m uvicorn src.api:app --reload
echo   3. Open http://localhost:8000/docs
echo.
