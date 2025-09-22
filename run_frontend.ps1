# Human Activity Recognition - Local Frontend Startup
# Run this script to start the frontend development server

Write-Host "🚀 Starting Human Activity Recognition Frontend..." -ForegroundColor Green

# Check if Node.js is available
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Node.js/npm is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Navigate to frontend directory
Set-Location "C:\ASH_PROJECT\frontend"

# Check if required files exist
if (-not (Test-Path "package.json")) {
    Write-Host "❌ package.json not found in frontend directory" -ForegroundColor Red
    exit 1
}

# Install dependencies if node_modules doesn't exist
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Start the frontend server
Write-Host "🌟 Starting frontend server..." -ForegroundColor Green
Write-Host "Backend should be running on http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    npm run web
} catch {
    Write-Host "❌ Failed to start frontend server" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}