param(
    [string]$Datasets = "Citeseer,DBLP,PubMed,amz_Photo,CoRA,Twitch_EN",
    [int]$Runs = 10,
    [int]$Epochs = 401,
    [int]$Patience = 20,
    [string]$Device = "cuda:0"
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
    $summaryName = "paper_baselines_$model.json"
    Write-Host "[run] env=$model model=$model"
    conda run -n $model --no-capture-output python "G:\myProject\ASLAM-3\benchmark_paper_baselines.py" `
        --models $model `
        --datasets $Datasets `
        --runs $Runs `
        --epochs $Epochs `
        --patience $Patience `
        --device $Device `
        --summary_name $summaryName
}

conda run -n ASLAM_THC --no-capture-output python "G:\myProject\ASLAM-3\compile_paper_baseline_report.py"
