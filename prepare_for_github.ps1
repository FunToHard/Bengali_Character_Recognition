# Prepare repository for GitHub by organizing and cleaning up files
$ErrorActionPreference = "Stop"

Write-Host "Preparing repository for GitHub..." -ForegroundColor Green

# Create necessary directories if they don't exist
@('data/train', 'data/val', 'models', 'images') | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
        Write-Host "Created directory: $_"
    }
}

# Keep only the latest model checkpoint
Write-Host "Organizing model checkpoints..."
$latestCheckpoint = Get-ChildItem "models/checkpoint_epoch_*.pth" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1
if ($latestCheckpoint) {
    # Only copy if it's not already the correct name
    if ($latestCheckpoint.Name -ne "checkpoint_epoch_49.pth") {
        Write-Host "Renaming latest checkpoint to checkpoint_epoch_49.pth..."
        Copy-Item $latestCheckpoint.FullName "models/checkpoint_epoch_49.pth" -Force
    } else {
        Write-Host "Latest checkpoint is already named correctly."
    }
    
    # Remove other checkpoints
    Get-ChildItem "models/checkpoint_epoch_*.pth" | 
        Where-Object { $_.Name -ne "checkpoint_epoch_49.pth" } | 
        ForEach-Object { 
            Write-Host "Removing old checkpoint: $($_.Name)"
            Remove-Item $_.FullName 
        }
}

# Create empty .gitkeep files
@('data/train/.gitkeep', 'data/val/.gitkeep') | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType File -Path $_ -Force | Out-Null
        Write-Host "Created: $_"
    }
}

# List of essential files that should be included
$essentialFiles = @(
    "bengali_gui.py",
    "data_loader.py",
    "evaluate.py",
    "guide.md",
    "model.py",
    "recognize_text.py",
    "requirements.txt",
    "run_recognition.bat",
    "train_model.ps1",
    "train.py",
    "utils.py",
    ".gitignore",
    "README.md",
    "models/checkpoint_epoch_49.pth",
    "images/gui_interface.png",
    "images/sample_predictions.png",
    "images/confusion_matrix.png",
    "images/training_history.png"
)

# Verify essential files exist
Write-Host "`nChecking essential files..."
$missingFiles = $essentialFiles | Where-Object { -not (Test-Path $_) }
if ($missingFiles) {
    Write-Host "`nWarning: The following essential files are missing:" -ForegroundColor Yellow
    $missingFiles | ForEach-Object { Write-Host "- $_" }
}

Write-Host "`nSetting up Git configuration..." -ForegroundColor Green
$email = Read-Host "Enter your GitHub email"
$name = Read-Host "Enter your GitHub username"

# Configure Git globally
Write-Host "Configuring Git..."
git config --global user.email $email
git config --global user.name $name

# Verify the configuration
Write-Host "`nGit configuration:" -ForegroundColor Green
Write-Host "Email: $(git config --global user.email)"
Write-Host "Name: $(git config --global user.name)"

Write-Host "`nSetting up Git repository..." -ForegroundColor Green

# Check if git is already initialized
if (Test-Path ".git") {
    Write-Host "Git repository already initialized" -ForegroundColor Yellow
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Cyan
    git init
}

# Check if remote origin exists and handle it
$remoteExists = git remote -v | Select-String "origin"
if ($remoteExists) {
    Write-Host "`nRemote 'origin' already exists. What would you like to do?" -ForegroundColor Yellow
    Write-Host "1. Remove existing and add new remote" -ForegroundColor White
    Write-Host "2. Keep existing remote" -ForegroundColor White
    $choice = Read-Host "Enter your choice (1 or 2)"
    
    if ($choice -eq "1") {
        Write-Host "Removing existing remote..." -ForegroundColor Cyan
        git remote remove origin
        $newRemote = Read-Host "Enter your new GitHub repository URL"
        Write-Host "Adding new remote..." -ForegroundColor Cyan
        git remote add origin $newRemote
    }
} else {
    $newRemote = Read-Host "Enter your GitHub repository URL"
    Write-Host "Adding remote..." -ForegroundColor Cyan
    git remote add origin $newRemote
}

Write-Host "`nRepository is ready for GitHub!" -ForegroundColor Green
Write-Host "You can now push your code with:" -ForegroundColor White
Write-Host "git add ." -ForegroundColor Cyan
Write-Host "git commit -m 'Initial commit'" -ForegroundColor Cyan
Write-Host "git branch -M main" -ForegroundColor Cyan
Write-Host "git push -u origin main" -ForegroundColor Cyan

Write-Host "`nPress any key to exit..."
pause
