# Helper script to pull the local Qwen model for Ollama
Write-Host "Checking Ollama status..." -ForegroundColor Cyan

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Host "Ollama is not in PATH. Checking default install directory..." -ForegroundColor Yellow
    $defaultOllama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $defaultOllama) {
        Write-Host "Found Ollama at: $defaultOllama" -ForegroundColor Green
        & $defaultOllama pull qwen3:4b
    } else {
        Write-Host "Ollama executable not found. Please install Ollama from https://ollama.com" -ForegroundColor Red
    }
} else {
    Write-Host "Pulling qwen3:4b..." -ForegroundColor Green
    ollama pull qwen3:4b
}
