# Training script for Bengali Character Recognition
$ErrorActionPreference = "Stop"

Write-Host "Activating Python 3.9 environment..." -ForegroundColor Green
& .\venv39\Scripts\Activate.ps1

Write-Host "Starting model training..." -ForegroundColor Green
Write-Host "This may take several hours. Progress will be displayed..." -ForegroundColor Yellow

# Training parameters
$env:CUDA_VISIBLE_DEVICES = "0"  # Use first GPU
python train.py --batch-size 128 --epochs 50 --learning-rate 0.001

Write-Host "`nTraining complete. Check the models folder for checkpoints." -ForegroundColor Green
pause
