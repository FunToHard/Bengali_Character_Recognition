@echo off
setlocal enabledelayedexpansion

:: Change to the script's directory
cd /d "%~dp0"

echo Activating Python 3.9 environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Python virtual environment not found.
    echo Please ensure you have run the setup script first.
    pause
    exit /b 1
)

:: Check if required packages are installed
python -c "import tkinterdnd2" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install tkinterdnd2
)

echo Starting Bengali Character Recognition GUI...
python bengali_gui.py
if errorlevel 1 (
    echo Error: Failed to start the application.
    echo Please check if all requirements are installed.
    pause
    exit /b 1
)

echo.
echo Press any key to exit...
pause > nul
