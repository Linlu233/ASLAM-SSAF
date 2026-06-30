param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [int]$Repeats = 10,
    [string]$ModelVariant = "ssmattn",
    [int]$Epochs = 401,
    [int]$Patience = 20,
    [int]$BatchSize = 32,
    [double]$Lr = 0.001,
    [double]$Wd = 0.0005,
    [string]$Cuda = "cuda:0",
    [string]$Root = "G:\myProject\ASLAM-3\datasets",
    [int]$MaxRetries = 3,
    [int]$RetrySleepSeconds = 15
)

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$codeRoot = "G:\myProject\ASLAM-3\ASLAM\code"
$pythonExe = "C:\Users\linlu\miniconda3\envs\ASLAM_THC\python.exe"
$resultRoot = "G:\myProject\ASLAM-3\ASLAM\results\train=0.9"

function Invoke-OneRun([int]$AttemptIndex) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $consoleLog = Join-Path $resultRoot "$($Dataset)_$($ModelVariant)_seriesrun$AttemptIndex`_$stamp.log"
    Set-Location $codeRoot
    & $pythonExe "train.py" `
        "--dataset" $Dataset `
        "--model_variant" $ModelVariant `
        "--runs" "1" `
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
    return $LASTEXITCODE
}

for ($i = 1; $i -le $Repeats; $i++) {
    $ok = $false
    for ($retry = 1; $retry -le $MaxRetries; $retry++) {
        $exitCode = Invoke-OneRun -AttemptIndex $i
        if ($exitCode -eq 0) {
            $ok = $true
            break
        }
        Start-Sleep -Seconds $RetrySleepSeconds
    }
    if (-not $ok) {
        throw "Dataset $Dataset failed at series run $i after $MaxRetries retries."
    }
}
