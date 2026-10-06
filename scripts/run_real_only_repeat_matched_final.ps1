param(
    [int[]]$Seeds = @(20260820, 20260821, 20260822),
    [string]$Device = "0",
    [string]$Data = "configs/dataset_repeat_matched_v1.yaml",
    [string]$ContentLock = "data/manifests/dataset_repeat_matched_trainval_lock_v1.json"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Missing .venv. Run .\scripts\bootstrap_real_only.ps1 first."
}

if (-not (Test-Path -LiteralPath $Data -PathType Leaf)) {
    throw "Missing relocated repeat-matched dataset YAML: $Data. Run bootstrap_repeat_matched.ps1 first."
}
if (-not (Test-Path -LiteralPath $ContentLock -PathType Leaf)) {
    throw "Missing relocated repeat-matched content lock: $ContentLock. Run bootstrap_repeat_matched.ps1 first."
}

foreach ($Seed in $Seeds) {
    $RunName = "real_only_repeat_matched_final_s${Seed}_100e"
    $RunDir = Join-Path $ProjectRoot "runs\yolo\$RunName"
    $MetricsPath = Join-Path $ProjectRoot "reports\yolo_eval\${RunName}_val\metrics.json"
    if (Test-Path -LiteralPath $MetricsPath -PathType Leaf) {
        Write-Host "Completed report already exists; skipping $RunName"
        continue
    }
    if (Test-Path -LiteralPath $RunDir) {
        throw "Partial or unverified run directory exists: $RunDir. Preserve it and inspect before retrying."
    }

    & $PythonExe "scripts\train_yolo.py" `
        --data $Data `
        --model "yolo11n.pt" `
        --epochs 100 `
        --imgsz 640 `
        --batch 16 `
        --device $Device `
        --workers 4 `
        --seed $Seed `
        --name $RunName `
        --optimizer SGD `
        --lr0 0.01 `
        --lrf 0.01 `
        --momentum 0.937 `
        --weight-decay 0.0005 `
        --warmup-epochs 3.0 `
        --amp `
        --content-lock $ContentLock `
        --content-lock-mode required
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunName" }

    & $PythonExe "scripts\evaluate_yolo.py" `
        --weights "runs/yolo/$RunName/weights/best.pt" `
        --data $Data `
        --split val `
        --content-lock $ContentLock `
        --content-lock-mode required `
        --imgsz 640 `
        --batch 16 `
        --device $Device `
        --workers 4 `
        --seed $Seed `
        --conf 0.001 `
        --iou 0.7 `
        --max-det 300 `
        --output-root "reports/yolo_eval" `
        --name "${RunName}_val"
    if ($LASTEXITCODE -ne 0) { throw "Independent validation failed: $RunName" }
}

$RequiredSeeds = @(20260820, 20260821, 20260822)
if (($Seeds.Count -eq $RequiredSeeds.Count) -and -not (Compare-Object $Seeds $RequiredSeeds)) {
    & $PythonExe "scripts\aggregate_real_only_repeat_matched.py"
    if ($LASTEXITCODE -ne 0) { throw "Aggregation failed." }
}
Write-Host "Requested Real-only-repeat-matched runs finished. Locked test data was not accessed."
