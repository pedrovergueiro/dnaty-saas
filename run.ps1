# dNATY SaaS — dev runner (PowerShell)
# Usage: from ANY directory: .\dnaty_saas\run.ps1
#        or from inside dnaty_saas\: .\run.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Create .env from example if it doesn't exist
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[setup] .env created from .env.example — edit it to set API_KEY"
}

Write-Host "[start] dNATY SaaS API → http://localhost:8000/docs"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
