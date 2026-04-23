# Tunnel-only launcher (non-intrusive)
# - Does NOT stop/restart backend/frontend
# - Detects active ports and opens localtunnel in separate terminals

$ErrorActionPreference = "Stop"

function Test-Http200 {
    param(
        [string]$Url
    )
    try {
        $r = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec 3
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Pick-BackendPort {
    $candidates = @(5000, 8000)
    foreach ($p in $candidates) {
        $listening = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue
        if ($listening -and (Test-Http200 -Url "http://localhost:$p/health")) {
            return $p
        }
    }
    return $null
}

function Pick-FrontendPort {
    # Prefer 8081 first because Vite auto-moved there in your current session
    $candidates = @(8081, 8080, 5173)
    foreach ($p in $candidates) {
        $listening = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue
        if ($listening -and (Test-Http200 -Url "http://localhost:$p")) {
            return $p
        }
    }
    return $null
}

$backendPort = Pick-BackendPort
$frontendPort = Pick-FrontendPort

if (-not $backendPort) {
    Write-Host "Backend not detected on 5000/8000 with healthy /health endpoint." -ForegroundColor Red
    Write-Host "Keep your backend running, then re-run this script." -ForegroundColor Yellow
    exit 1
}

if (-not $frontendPort) {
    Write-Host "Frontend not detected on 8081/8080/5173." -ForegroundColor Red
    Write-Host "Keep Vite running, then re-run this script." -ForegroundColor Yellow
    exit 1
}

Write-Host "Detected backend port: $backendPort" -ForegroundColor Green
Write-Host "Detected frontend port: $frontendPort" -ForegroundColor Green

# Open 2 new PowerShell windows for tunnels so they stay alive
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npx localtunnel --port $backendPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npx localtunnel --port $frontendPort"

Write-Host "Opened two tunnel windows." -ForegroundColor Cyan
Write-Host "In each tunnel window, copy the generated https://*.loca.lt URL." -ForegroundColor Cyan
Write-Host "Use the frontend tunnel URL in browser." -ForegroundColor Cyan
Write-Host "Use the backend tunnel URL in API config if needed." -ForegroundColor Cyan
