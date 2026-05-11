cargo build --release -p brain-napi
if ($LASTEXITCODE -eq 0) {
    Copy-Item target\release\brain_napi.dll brain-napi\brain-napi.node -Force
    Write-Host "brain-napi.node updated" -ForegroundColor Green
}
