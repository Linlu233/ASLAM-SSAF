param(
    [string]$TargetRoot = "G:\myProject\ASLAM-3\external\official_baselines",
    [string]$Proxy = "http://127.0.0.1:10809",
    [switch]$UseProxy
)

$ErrorActionPreference = "Stop"

if ($UseProxy) {
    $env:http_proxy = $Proxy
    $env:https_proxy = $Proxy
    Write-Host "[proxy] using $Proxy"
}

$repos = @{
    "pytorch_geometric" = "https://github.com/pyg-team/pytorch_geometric.git"
    "SuperGAT" = "https://github.com/dongkwan-kim/SuperGAT.git"
    "SEAL" = "https://github.com/muhanzhang/SEAL.git"
    "SEAL_OGB" = "https://github.com/facebookresearch/SEAL_OGB.git"
}

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

foreach ($item in $repos.GetEnumerator()) {
    $dst = Join-Path $TargetRoot $item.Key
    if (Test-Path $dst) {
        Write-Host "[skip] $($item.Key) already exists"
        continue
    }
    Write-Host "[clone] $($item.Value) -> $dst"
    git clone $item.Value $dst
}
