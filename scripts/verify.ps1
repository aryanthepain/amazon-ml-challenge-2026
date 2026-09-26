# Deterministic Verification Gate for Amazon ML Challenge 2026 Phase 1
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AMAZON ML CHALLENGE 2026: PHASE 1 VERIFICATION GATE  " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Run Foundation Pytest Suite
Write-Host "`n[1/4] Running Phase 1 Foundation Pytest Suite..." -ForegroundColor Yellow
pytest tests/test_phase1_foundation.py -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Foundation pytest suite failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Foundation pytest suite passed." -ForegroundColor Green

# 2. Run Kaggle Environment Pytest Suite
Write-Host "`n[2/4] Running Kaggle Remote Environment Pytest Suite..." -ForegroundColor Yellow
pytest tests/test_kaggle_env.py -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Kaggle environment pytest suite failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Kaggle environment pytest suite passed." -ForegroundColor Green

# 3. Check Data Loader Ingestion Diagnostic
Write-Host "`n[3/4] Checking Data Loader Ingestion Diagnostic..." -ForegroundColor Yellow
python src/data_loader.py --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Data loader diagnostic failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Ingestion & dynamic path resolution verified." -ForegroundColor Green

# 4. Check Kaggle Pre-Flight Doctor
Write-Host "`n[4/4] Checking Kaggle Credentials Doctor..." -ForegroundColor Yellow
python scripts/verify_kaggle_env.py --check-only
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Kaggle pre-flight doctor check failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Kaggle pre-flight credentials verified." -ForegroundColor Green

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host " [SUCCESS] All Phase 1 Verification Gates PASSED!     " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
exit 0
