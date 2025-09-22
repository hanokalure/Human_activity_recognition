# Human Activity Recognition - Hugging Face Spaces Deployment Script
# This script deploys the backend to Hugging Face Spaces

Write-Host "🚀 Deploying Human Activity Recognition Backend to Hugging Face Spaces..." -ForegroundColor Green

# Configuration
$HF_USERNAME = "Hanokalure"
$SPACE_NAME = "human-activity-backend"
$HF_SPACE_REPO = "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

# Check if git is available
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if hf_backend directory exists
if (-not (Test-Path "C:\ASH_PROJECT\hf_backend")) {
    Write-Host "❌ hf_backend directory not found" -ForegroundColor Red
    exit 1
}

# Navigate to hf_backend directory
Set-Location "C:\ASH_PROJECT\hf_backend"

# Initialize git repository if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "📦 Initializing git repository..." -ForegroundColor Yellow
    git init
    git config user.name "$HF_USERNAME"
    git config user.email "$HF_USERNAME@users.noreply.huggingface.co"
}

# Add Hugging Face remote if not exists
$remoteExists = git remote | Select-String "origin"
if (-not $remoteExists) {
    Write-Host "🔗 Adding Hugging Face Spaces remote..." -ForegroundColor Yellow
    git remote add origin $HF_SPACE_REPO
}

# Create .gitignore if it doesn't exist
if (-not (Test-Path ".gitignore")) {
    Write-Host "📝 Creating .gitignore..." -ForegroundColor Yellow
    @"
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8
}

# Stage all files
Write-Host "📂 Staging files for commit..." -ForegroundColor Yellow
git add .

# Commit changes
$commitMessage = "Deploy Human Activity Recognition backend to Hugging Face Spaces"
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
git commit -m $commitMessage

# Push to Hugging Face Spaces
Write-Host "🚀 Pushing to Hugging Face Spaces..." -ForegroundColor Yellow
Write-Host "Note: You'll be prompted for your Hugging Face token" -ForegroundColor Cyan

try {
    git push -u origin main
    Write-Host "✅ Successfully deployed to Hugging Face Spaces!" -ForegroundColor Green
    Write-Host "🌐 Your backend will be available at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
    Write-Host "🔗 API Base URL: https://$HF_USERNAME-$SPACE_NAME.hf.space" -ForegroundColor Cyan
    
    # Instructions for next steps
    Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Wait for the Space to build (usually 2-5 minutes)" -ForegroundColor White
    Write-Host "2. Test the Gradio interface at the Space URL" -ForegroundColor White
    Write-Host "3. Update your Vercel frontend to use the new API URL" -ForegroundColor White
    Write-Host "4. Test the full integration" -ForegroundColor White
} catch {
    Write-Host "❌ Failed to push to Hugging Face Spaces" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`n💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure you have a Hugging Face account" -ForegroundColor White
    Write-Host "2. Create a write-access token at: https://huggingface.co/settings/tokens" -ForegroundColor White
    Write-Host "3. Create the Space manually at: https://huggingface.co/new-space" -ForegroundColor White
    Write-Host "4. Use your HF token as password when prompted" -ForegroundColor White
}

# Return to original directory
Set-Location "C:\ASH_PROJECT"

Write-Host "`n🎉 Deployment script completed!" -ForegroundColor Green