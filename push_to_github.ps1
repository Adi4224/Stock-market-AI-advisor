# ==============================================================================
# Stock Market AI Advisor - Push to GitHub Helper
# Developed by: Adithya Dadi
# ==============================================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Stock Market AI Advisor - GitHub Initializer & Pusher     " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Get GitHub repository URL from the user
$repoUrl = Read-Host -Prompt "Enter your GitHub Repository URL (e.g., https://github.com/username/repo.git)"

if (-not $repoUrl) {
    Write-Host "Error: Repository URL cannot be empty." -ForegroundColor Red
    Exit
}

# Verify git repository exists
if (-not (Test-Path .git)) {
    Write-Host "Initializing local Git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
}

# Link remote origin
Write-Host "Setting up remote origin to $repoUrl..." -ForegroundColor Yellow
# Remove existing remote if present
git remote remove origin 2>$null
git remote add origin $repoUrl

# Stage and commit files conforming to .gitignore
Write-Host "Staging files..." -ForegroundColor Yellow
git add .

Write-Host "Creating commit..." -ForegroundColor Yellow
git commit -m "Initial commit - Stock Market AI Advisor by Adithya Dadi" 2>$null

# Push to remote using local user credentials
Write-Host "Pushing project code to GitHub..." -ForegroundColor Yellow
Write-Host "This will securely leverage your local Windows Git credentials." -ForegroundColor Gray
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n============================================================" -ForegroundColor Green
    Write-Host "  SUCCESS: Project successfully pushed to GitHub!           " -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host "`nPush failed. Please ensure the repository exists and you have permission." -ForegroundColor Red
}

Read-Host -Prompt "Press Enter to exit..."
