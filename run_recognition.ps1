# Run Bengali Character Recognition GUI
$ErrorActionPreference = "Stop"

Write-Host "Activating Python 3.9 environment..." -ForegroundColor Green

# Ensure we're in the correct directory
Set-Location -Path $PSScriptRoot

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: Python virtual environment not found." -ForegroundColor Red
    Write-Host "Please ensure you have run the setup script first." -ForegroundColor Yellow
    pause
    exit 1
}

# Check if required packages are installed
try {
    python -c "import tkinterdnd2"
} catch {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install tkinterdnd2
}

Write-Host "Starting Bengali Character Recognition GUI..." -ForegroundColor Green
try {
    python bengali_gui.py
} catch {
    Write-Host "Error: Failed to start the application." -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host "Please check if all requirements are installed." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "`nPress any key to exit..." -ForegroundColor Cyan
pause
