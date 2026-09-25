# flex_search launcher (VPS). One command: checks the environment, then runs the whole staged search
# (stage1 -> stage2 -> stage3 -> stage4 + 2016-2023 stress) in ONE hidden, BelowNormal-priority Python process
# that uses a 4-worker pool. Resumable: re-running this script skips stage-1 work already saved in results\.
#
#   cd C:\Users\Administrator\MeanReversion\flex_search
#   .\start_flex_search.ps1                 # default settings
#   .\start_flex_search.ps1 -N1 40000       # more stage-1 configs per session
#
# Progress:   Get-Content .\results\run.log -Tail 20 -Wait
# Stop:       Stop-Process -Id <PID printed at start> (the pool workers exit with it); re-run this script to resume
param(
    [int]$Workers = 4,
    [int]$N1 = 30000,
    [int]$Top2 = 300,
    [int]$NN = 16,
    [int]$K3 = 12,
    [int]$R3 = 12,
    [int]$Top4 = 60
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$data = Join-Path $here "..\NASDAQFuturesData\nq_continuous_1min.parquet"
if (-not $env:MR_NQ_PARQUET -and -not (Test-Path $data)) {
    Write-Host "Missing data file: $data" -ForegroundColor Red
    Write-Host "Copy NASDAQFuturesData\nq_continuous_1min.parquet from the laptop (it is not in git), or set `$env:MR_NQ_PARQUET." -ForegroundColor Red
    exit 1
}
$py = (Get-Command py -ErrorAction SilentlyContinue)
$exe = if ($py) { "py" } else { "python" }
$pre = if ($py) { @("-3") } else { @() }
Write-Host "Checking Python packages (numpy, pandas, pyarrow, tzdata)..."
& $exe @pre -m pip install --quiet numpy pandas pyarrow tzdata
New-Item -ItemType Directory -Force -Path (Join-Path $here "logs") | Out-Null
$pyArgs = $pre + @("fs_search.py", "all", "--workers", $Workers, "--n1", $N1, "--top2", $Top2, "--nn", $NN,
                 "--k3", $K3, "--r3", $R3, "--top4", $Top4, "--stress")
$p = Start-Process -FilePath $exe -ArgumentList $pyArgs -WorkingDirectory $here -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $here "logs\stdout.log") -RedirectStandardError (Join-Path $here "logs\stderr.log")
Start-Sleep -Seconds 2
try { $p.PriorityClass = "BelowNormal" } catch {}
Write-Host "Started flex_search (PID $($p.Id)) with $Workers workers, N1=$N1 per session." -ForegroundColor Green
Set-Content -Path (Join-Path $here "logs\pid.txt") -Value $p.Id
Write-Host "Progress:  Get-Content .\results\run.log -Tail 20 -Wait"
Write-Host "Finished when .\results\DONE exists. Then commit results: git add results; git commit -m 'flex_search results'; git push"
