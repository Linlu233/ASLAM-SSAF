param(
    [string]$ModelVariant = "ssmattn",
    [int]$Runs = 10,
    [int]$Epochs = 401,
    [int]$Patience = 20,
    [int]$BatchSize = 32,
    [double]$Lr = 0.001,
    [double]$Wd = 0.0005,
    [string]$Cuda = "cuda:0",
    [string]$Root = "G:\myProject\ASLAM-3\datasets"
)

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$codeRoot = "G:\myProject\ASLAM-3\ASLAM\code"
$pythonExe = "C:\Users\linlu\miniconda3\envs\ASLAM_THC\python.exe"
$resultRoot = "G:\myProject\ASLAM-3\ASLAM\results\train=0.9"

function Run-Dataset([string]$Dataset) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $consoleLog = Join-Path $resultRoot "$($Dataset)_$($ModelVariant)_console_$stamp.log"

    Set-Location $codeRoot
    & $pythonExe "train.py" `
        "--dataset" $Dataset `
        "--model_variant" $ModelVariant `
        "--runs" $Runs `
        "--epochs" $Epochs `
        "--patience" $Patience `
        "--bs" $BatchSize `
        "--lr" $Lr `
        "--wd" $Wd `
        "--train_percent" "1.0" `
        "--val_percent" "1.0" `
        "--test_percent" "1.0" `
        "--root" $Root `
        "--cuda" $Cuda *> $consoleLog

    if ($LASTEXITCODE -ne 0) {
        throw "Dataset $Dataset failed with exit code $LASTEXITCODE"
    }
}

Run-Dataset "photo"
Run-Dataset "pubmed"
