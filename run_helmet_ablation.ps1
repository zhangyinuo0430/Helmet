param(
    [string]$Data = "ultralytics/cfg/datasets/helmet_42_yolo_dataset.yaml",
    [int]$Epochs = 100,
    [int]$ImgSize = 640,
    [int]$Batch = 4,
    [string]$Device = "0",
    [int]$Workers = 0,
    [string]$Project = "runs/helmet_ablation",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# 设置编码为UTF-8
$OutputEncoding = [Console]::OutputEncoding = [Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "UTF-8"

$env:PYTHONPATH = "$Root;$env:PYTHONPATH"

$experiments = @(
    # @{ Name = "01_baseline_yolov8"; Model = "ultralytics/cfg/models/v8/helmet-yolov8-baseline.yaml"; Extra = @() }
    # @{ Name = "02_spd_conv"; Model = "ultralytics/cfg/models/v8/helmet-yolov8-spd.yaml"; Extra = @() }
    # @{ Name = "03_p2_layer"; Model = "ultralytics/cfg/models/v8/helmet-yolov8-p2.yaml"; Extra = @() }
    # @{ Name = "04_ca"; Model = "ultralytics/cfg/models/v8/helmet-yolov8-ca.yaml"; Extra = @() }
    @{ Name = "05_osa_occlusion_attention"; Model = "ultralytics/cfg/models/v8/helmet-yolov8-osa.yaml"; Extra = @("copy_paste=0.15", "erasing=0.35") }
    # @{ Name = "06_full_spd_p2_ca_osa"; Model = "ultralytics/cfg/models/v8/helmet-yolov8-full.yaml"; Extra = @("copy_paste=0.15", "erasing=0.35") }
)

Write-Host "YOLOv8 helmet ablation experiments" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host "Data: $Data"
Write-Host "Epochs: $Epochs, ImgSize: $ImgSize, Batch: $Batch, Device: $Device"
Write-Host "Project: $Project"

foreach ($exp in $experiments) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor DarkGray
    Write-Host "Running $($exp.Name)" -ForegroundColor Green
    Write-Host "Model: $($exp.Model)"

    $args = @(
        "detect", "train",
        "model=$($exp.Model)",
        "data=$Data",
        "epochs=$Epochs",
        "imgsz=$ImgSize",
        "batch=$Batch",
        "device=$Device",
        "workers=$Workers",
        "project=$Project",
        "name=$($exp.Name)",
        "exist_ok=True",
        "seed=42",
        "deterministic=True",
        "pretrained=False",
        "cache=False",
        "amp=True",
        "close_mosaic=10",
        "mosaic=1.0",
        "mixup=0.10",
        "hsv_h=0.015",
        "hsv_s=0.70",
        "hsv_v=0.40",
        "degrees=5.0",
        "translate=0.10",
        "scale=0.75",
        "fliplr=0.50",
        "plots=True",
        "val=True"
    ) + $exp.Extra

    if ($DryRun) {
        Write-Host "python train_runner.py $($exp.Model) $Data --device $Device --epochs $Epochs --batch $Batch --imgsz $ImgSize --project $Project --name $($exp.Name) $($exp.Extra -join ' ')" -ForegroundColor Yellow
    } else {
        python train_runner.py $exp.Model $Data --device $Device --epochs $Epochs --batch $Batch --imgsz $ImgSize --project $Project --name $exp.Name $exp.Extra
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $($exp.Name) failed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host ""
Write-Host "All ablation experiments finished. Results are in: $Project" -ForegroundColor Cyan
