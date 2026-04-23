# Expose Backend API to Internet
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/c npx localtunnel --port 5000 --subdomain rag-backend-api"

# Wait a few seconds for backend tunnel to stabilize
Start-Sleep -Seconds 5

# Start Frontend Dev Server locally
cd frontend
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/c npm run dev"

# Wait a few seconds for vite to boot
Start-Sleep -Seconds 5

# Expose Frontend to Internet
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/c npx localtunnel --port 8080 --subdomain rag-project-ui"

Write-Host "========================================="
Write-Host "Services exposed to the internet live:"
Write-Host "Backend API: https://rag-backend-api.loca.lt"
Write-Host "Frontend UI: https://rag-project-ui.loca.lt"
Write-Host "========================================="
Write-Host "To close all tunnels, press CTRL+C and run: Stop-Process -Name 'node' -Force"
