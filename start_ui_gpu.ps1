$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
$Python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
$Port = 7860
$Url = "http://127.0.0.1:$Port/"
$Recognizer = Join-Path $ProjectDir "runs\crnn_plate_balanced\best.pt"
$Log = Join-Path $ProjectDir "runs\ui_server.log"
$Err = Join-Path $ProjectDir "runs\ui_server.err.log"

Set-Location $ProjectDir
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "runs") | Out-Null

Write-Host "Starting YOLO 26 + CRNN UI..."
Write-Host "Project: $ProjectDir"
Write-Host "Python: $Python"
Write-Host "URL: $Url"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python was not found: $Python"
    exit 1
}

$listeners = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
foreach ($listener in $listeners) {
    Write-Host "Closing old UI process on port ${Port}: $($listener.OwningProcess)"
    Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
}

Remove-Item -LiteralPath $Log, $Err -Force -ErrorAction SilentlyContinue

$args = @(
    "-u",
    "scripts\ui_app.py",
    "--device", "cuda",
    "--port", "$Port",
    "--recognizer", $Recognizer
)

Start-Process -FilePath $Python `
    -ArgumentList $args `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $Log `
    -RedirectStandardError $Err `
    -WindowStyle Minimized

Write-Host "Waiting for service to be ready..."
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/metrics" -UseBasicParsing -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host ""
    Write-Host "UI service failed to start. Error log:"
    if (Test-Path $Err) {
        Get-Content -LiteralPath $Err
    } else {
        Write-Host "No error log was created."
    }
    exit 1
}

Write-Host "UI is ready. Opening browser..."
Start-Process $Url
Write-Host ""
Write-Host "If the browser did not open, visit: $Url"
Write-Host "You can close this window. The UI server is running in the background."
