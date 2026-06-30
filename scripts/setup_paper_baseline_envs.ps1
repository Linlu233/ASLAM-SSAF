param(
    [string]$BaseEnv = "ASLAM_THC",
    [string]$CondaRoot = "$env:USERPROFILE\miniconda3"
)

$ErrorActionPreference = "Stop"
$env:CONDA_NO_PLUGINS = "true"

$models = @(
    "cn",
    "aa",
    "node2vec",
    "gcn",
    "gae",
    "vgae",
    "graphsage",
    "gat",
    "supergat",
    "seal",
    "bsal"
)

foreach ($model in $models) {
    $envPath = Join-Path $CondaRoot "envs\$model"
    if (Test-Path $envPath) {
        Write-Host "[skip] env '$model' already exists at $envPath"
        continue
    }

    Write-Host "[create] cloning '$BaseEnv' -> '$model'"
    conda create --name $model --clone $BaseEnv -y
    conda run -n $model --no-capture-output python -c "import torch; print('env=$model torch=' + torch.__version__); print('cuda=' + str(torch.cuda.is_available()))"
}
