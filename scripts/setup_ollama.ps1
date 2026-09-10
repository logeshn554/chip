# Helper script to pull the local Qwen model for Ollama (Qwen-14B)
param([string]$Model = "qwen2.5:14b")
Write-Host "Checking Ollama status for model $Model..." -ForegroundColor Cyan

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Host "Ollama is not in PATH. Checking default install directory..." -ForegroundColor Yellow
    $defaultOllama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $defaultOllama) {
        Write-Host "Found Ollama at: $defaultOllama" -ForegroundColor Green
        & $defaultOllama pull $Model
    } else {
        Write-Host "Ollama executable not found. Please install Ollama from https://ollama.com" -ForegroundColor Red
    }
} else {
    Write-Host "Pulling $Model..." -ForegroundColor Green
    ollama pull $Model
}
