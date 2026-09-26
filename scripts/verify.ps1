# Deterministic Verification Gate for Amazon ML Challenge 2026 Phase 1
$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AMAZON ML CHALLENGE 2026: PHASE 1 VERIFICATION GATE  " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Run pytest suite
Write-Host "`n[1/3] Running Phase 1 Foundation Pytest Suite..." -ForegroundColor Yellow
$pytestOutput = pytest tests/test_phase1_foundation.py -v --tb=short
$pytestExit = $LASTEXITCODE

if ($pytestExit -eq 0) {
    Write-Host "[PASS] All Phase 1 foundation tests passed successfully." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Pytest encountered failures (Exit code: $pytestExit)." -ForegroundColor Red
    exit $pytestExit
}

# 2. Check Data Loader diagnostic
Write-Host "`n[2/3] Checking Data Loader Ingestion Diagnostic..." -ForegroundColor Yellow
python src/data_loader.py --check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Data loader diagnostic failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Ingestion & dynamic path resolution verified." -ForegroundColor Green

# 3. Check Evaluator Help
Write-Host "`n[3/3] Checking Evaluation Harness CLI..." -ForegroundColor Yellow
python evaluate.py --help > $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] evaluate.py CLI failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] evaluate.py verified." -ForegroundColor Green

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host " [SUCCESS] All Phase 1 Foundation Gates PASSED!       " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
exit 0
