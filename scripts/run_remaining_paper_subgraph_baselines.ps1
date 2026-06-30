param(
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$env:CONDA_NO_PLUGINS = "true"
$env:PYTHONWARNINGS = "ignore"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$projectRoot = "G:\myProject\ASLAM-3"
$benchmark = Join-Path $projectRoot "benchmark_paper_baselines.py"
$compiler = Join-Path $projectRoot "compile_paper_baseline_report.py"
$scriptLog = Join-Path $projectRoot "results\paper_baselines\remaining_subgraph_suite.script.log"

Add-Content -Path $scriptLog -Value ("[script-start] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

function Test-TaskComplete {
    param(
        [string]$SummaryName,
        [string]$Dataset,
        [string]$ModelName,
        [int]$Seed,
        [int]$Runs
    )

    $summaryPath = Join-Path $projectRoot ("results\paper_baselines\" + $SummaryName)
    if (-not (Test-Path $summaryPath)) {
        return $false
    }

    try {
        $payload = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
        $datasetPayload = $payload.datasets.$Dataset
        if ($null -eq $datasetPayload) {
            return $false
        }
        $modelPayload = $datasetPayload.$ModelName
        if ($null -eq $modelPayload) {
            return $false
        }
        $doneSeeds = @()
        foreach ($run in $modelPayload.runs) {
            $doneSeeds += [int]$run.seed
        }
        foreach ($offset in 0..($Runs - 1)) {
            if (($doneSeeds -contains ($Seed + $offset)) -eq $false) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Invoke-BenchmarkChunk {
    param(
        [string]$EnvName,
        [string]$ModelName,
        [string]$Dataset,
        [int]$Seed,
        [int]$Runs,
        [string]$SummaryName
    )

    if (Test-TaskComplete -SummaryName $SummaryName -Dataset $Dataset -ModelName $ModelName -Seed $Seed -Runs $Runs) {
        Add-Content -Path $scriptLog -Value ("[skip] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " env=$EnvName model=$ModelName dataset=$Dataset seed=$Seed runs=$Runs")
        Write-Host "[skip] env=$EnvName model=$ModelName dataset=$Dataset seed=$Seed runs=$Runs"
        return
    }

    Add-Content -Path $scriptLog -Value ("[run] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " env=$EnvName model=$ModelName dataset=$Dataset seed=$Seed runs=$Runs")
    Write-Host "[run] env=$EnvName model=$ModelName dataset=$Dataset seed=$Seed runs=$Runs"
    conda run -n $EnvName --no-capture-output python $benchmark `
        --datasets $Dataset `
        --models $ModelName `
        --seed $Seed `
        --runs $Runs `
        --epochs 401 `
        --patience 20 `
        --device $Device `
        --summary_name $SummaryName

    Add-Content -Path $scriptLog -Value ("[compile] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host "[compile] refresh partial report"
    conda run -n ASLAM_THC --no-capture-output python $compiler `
        --inputs "results/paper_baselines/paper_baselines_*.json" `
        --output "results/paper_baselines/paper_baseline_partial_report.md"
}

$tasks = @(
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "Twitch_EN"; Seed = 4; Runs = 2; SummaryName = "paper_baselines_seal_twitch_en.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "Twitch_EN"; Seed = 6; Runs = 2; SummaryName = "paper_baselines_seal_twitch_en.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "Twitch_EN"; Seed = 8; Runs = 2; SummaryName = "paper_baselines_seal_twitch_en.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "Twitch_EN"; Seed = 10; Runs = 2; SummaryName = "paper_baselines_seal_twitch_en.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "amz_Photo"; Seed = 2; Runs = 2; SummaryName = "paper_baselines_seal_amz_photo.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "amz_Photo"; Seed = 4; Runs = 2; SummaryName = "paper_baselines_seal_amz_photo.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "amz_Photo"; Seed = 6; Runs = 2; SummaryName = "paper_baselines_seal_amz_photo.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "amz_Photo"; Seed = 8; Runs = 2; SummaryName = "paper_baselines_seal_amz_photo.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "amz_Photo"; Seed = 10; Runs = 2; SummaryName = "paper_baselines_seal_amz_photo.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "PubMed"; Seed = 2; Runs = 2; SummaryName = "paper_baselines_seal_pubmed.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "PubMed"; Seed = 4; Runs = 2; SummaryName = "paper_baselines_seal_pubmed.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "PubMed"; Seed = 6; Runs = 2; SummaryName = "paper_baselines_seal_pubmed.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "PubMed"; Seed = 8; Runs = 2; SummaryName = "paper_baselines_seal_pubmed.json" }
    @{ EnvName = "seal"; ModelName = "seal"; Dataset = "PubMed"; Seed = 10; Runs = 2; SummaryName = "paper_baselines_seal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 2; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 3; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 4; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 5; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 6; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 7; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 8; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 9; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 10; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Citeseer"; Seed = 11; Runs = 1; SummaryName = "paper_baselines_bsal_citeseer.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 2; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 3; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 4; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 5; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 6; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 7; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 8; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 9; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 10; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "DBLP"; Seed = 11; Runs = 1; SummaryName = "paper_baselines_bsal_dblp.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 2; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 3; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 4; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 5; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 6; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 7; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 8; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 9; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 10; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "PubMed"; Seed = 11; Runs = 1; SummaryName = "paper_baselines_bsal_pubmed.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 2; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 3; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 4; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 5; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 6; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 7; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 8; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 9; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 10; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "amz_Photo"; Seed = 11; Runs = 1; SummaryName = "paper_baselines_bsal_amz_photo.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 2; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 3; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 4; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 5; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 6; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 7; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 8; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 9; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 10; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "CoRA"; Seed = 11; Runs = 1; SummaryName = "paper_baselines_bsal_cora.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 2; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 3; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 4; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 5; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 6; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 7; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 8; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 9; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 10; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
    @{ EnvName = "bsal"; ModelName = "bsal"; Dataset = "Twitch_EN"; Seed = 11; Runs = 1; SummaryName = "paper_baselines_bsal_twitch_en.json" }
)

foreach ($task in $tasks) {
    Invoke-BenchmarkChunk `
        -EnvName $task.EnvName `
        -ModelName $task.ModelName `
        -Dataset $task.Dataset `
        -Seed $task.Seed `
        -Runs $task.Runs `
        -SummaryName $task.SummaryName
}

Add-Content -Path $scriptLog -Value ("[done] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host "[done] remaining subgraph baseline queue finished"
