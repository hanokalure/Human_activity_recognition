# Human Activity Recognition - Local Backend Startup
# Run this script to start the backend server locally

Write-Host "🚀 Starting Human Activity Recognition Backend..." -ForegroundColor Green

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Navigate to backend directory
Set-Location "C:\ASH_PROJECT\hf_backend"

# Check if required files exist
if (-not (Test-Path "app.py")) {
    Write-Host "❌ app.py not found in hf_backend directory" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ requirements.txt not found" -ForegroundColor Red
    exit 1
}

# Install/update dependencies
Write-Host "📦 Installing/updating dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Start the backend server
Write-Host "🌟 Starting backend server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    python app.py
} catch {
    Write-Host "❌ Failed to start backend server" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}