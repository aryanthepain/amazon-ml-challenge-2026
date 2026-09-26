# Deterministic Verification Gate for Amazon ML Challenge 2026
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AMAZON ML CHALLENGE 2026: DETERMINISTIC TEST GATE    " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Run Foundation Pytest Suite
Write-Host "`n[1/5] Running Phase 1 Foundation Pytest Suite..." -ForegroundColor Yellow
pytest tests/test_phase1_foundation.py -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Foundation pytest suite failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Foundation pytest suite passed." -ForegroundColor Green

# 2. Run Kaggle Environment Pytest Suite
Write-Host "`n[2/5] Running Kaggle Remote Environment Pytest Suite..." -ForegroundColor Yellow
pytest tests/test_kaggle_env.py -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Kaggle environment pytest suite failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Kaggle environment pytest suite passed." -ForegroundColor Green

# 3. Run Phase 2 Text Preprocessing Pytest Suite
Write-Host "`n[3/5] Running Phase 2 Preprocessing Pytest Suite..." -ForegroundColor Yellow
pytest tests/test_phase2_preprocessing.py -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Phase 2 Preprocessing pytest suite failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Phase 2 Preprocessing pytest suite passed." -ForegroundColor Green

# 4. Check Data Loader Ingestion Diagnostic
Write-Host "`n[4/5] Checking Data Loader Ingestion Diagnostic..." -ForegroundColor Yellow
python src/data_loader.py --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Data loader diagnostic failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Ingestion & dynamic path resolution verified." -ForegroundColor Green

# 5. Check Kaggle Pre-Flight Doctor
Write-Host "`n[5/5] Checking Kaggle Credentials Doctor..." -ForegroundColor Yellow
python scripts/verify_kaggle_env.py --check-only
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Kaggle pre-flight doctor check failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Kaggle pre-flight credentials verified." -ForegroundColor Green

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host " [SUCCESS] All Deterministic Verification Gates PASSED! " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
exit 0
