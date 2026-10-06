param([switch]$SkipInstall)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

if (-not $SkipInstall) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $Created = $false
        foreach ($Version in @("3.11", "3.12", "3.10")) {
            & py "-$Version" -m venv ".venv"
            if ($LASTEXITCODE -eq 0) {
                $Created = $true
                break
            }
        }
        if (-not $Created) { throw "Python 3.10-3.12 is required." }
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv ".venv"
    } else {
        throw "Python 3.10-3.12 is required. Install Python and rerun this script."
    }
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Virtual environment is unavailable: $PythonExe"
}
if (-not $SkipInstall) {
    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $PythonExe -m pip install -r "requirements-real-only.txt"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    & $PythonExe -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }
}

& $PythonExe "scripts\verify_transfer_bundle.py"
if ($LASTEXITCODE -ne 0) { throw "Transfer bundle verification failed." }
& $PythonExe "scripts\prepare_real_only_machine.py"
if ($LASTEXITCODE -ne 0) { throw "Dataset relocation/content lock failed." }
& $PythonExe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
& $PythonExe -c "import json, torch, ultralytics; assert torch.cuda.is_available(), 'CUDA unavailable'; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'ultralytics':ultralytics.__version__}, ensure_ascii=False))"
if ($LASTEXITCODE -ne 0) { throw "CUDA/runtime verification failed." }
Write-Host "Bootstrap complete. Run .\scripts\run_real_only_final.ps1 to start all three seeds."
