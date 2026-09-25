# flex_search launcher (VPS). One command: checks the environment, then runs the whole staged search
# (stage1 -> stage2 -> stage3 -> stage4 + 2016-2023 stress) in ONE hidden, BelowNormal-priority Python process
# that uses a 4-worker pool. Resumable: re-running this script skips stage-1 work already saved.
#
#   cd C:\Users\Administrator\MeanReversion\flex_search
#   .\start_flex_search.ps1                                  # FundedNext Flex on NQ (results\)
#   .\start_flex_search.ps1 -Market cfd -Challenge ftmo2     # FTMO 2-step on the NDX100 CFD (results_ftmo2_cfd\)
#
# Stop:  Stop-Process -Id <PID printed at start>; re-run this script to resume
param(
    [ValidateSet("nq", "cfd")] [string]$Market = "nq",
    [ValidateSet("flex", "ftmo2")] [string]$Challenge = "flex",
    [int]$Workers = 4,
    [int]$N1 = 30000,
    [int]$Top2 = 300,
    [int]$NN = 16,
    [int]$K3 = 12,
    [int]$R3 = 12,
    [int]$Top4 = 100
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$env:FS_MARKET = $Market; $env:FS_CHALLENGE = $Challenge
$data = if ($Market -eq "cfd") { Join-Path $here "..\CFDData\ndx100_dukascopy_1min.parquet" } else { Join-Path $here "..\NASDAQFuturesData\nq_continuous_1min.parquet" }
if (-not $env:MR_NQ_PARQUET -and -not $env:MR_CFD_PARQUET -and -not (Test-Path $data)) {
    Write-Host "Missing data file: $data" -ForegroundColor Red
    Write-Host "Copy the data file from the laptop to that path (data is not in git)." -ForegroundColor Red
    exit 1
}
# Use the Windows "py" launcher (plain "python" is the Microsoft Store stub on this VPS).
Write-Host "Using: $(py -3 --version 2>&1)"
Write-Host "Checking Python packages (numpy, pandas, pyarrow, tzdata)..."
py -3 -m pip install --quiet numpy pandas pyarrow tzdata
New-Item -ItemType Directory -Force -Path (Join-Path $here "logs") | Out-Null
$argLine = "-3 fs_search.py all --workers $Workers --n1 $N1 --top2 $Top2 --nn $NN --k3 $K3 --r3 $R3 --top4 $Top4 --stress"
$p = Start-Process -FilePath "py" -ArgumentList $argLine -WorkingDirectory $here -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $here "logs\stdout.log") -RedirectStandardError (Join-Path $here "logs\stderr.log")
Start-Sleep -Seconds 2
try { $p.PriorityClass = "BelowNormal" } catch {}
Write-Host "Started flex_search (PID $($p.Id)) with $Workers workers, N1=$N1 per session." -ForegroundColor Green
Set-Content -Path (Join-Path $here "logs\pid.txt") -Value $p.Id
$resDir = if ($Market -eq "nq" -and $Challenge -eq "flex") { "results" } else { "results_$($Challenge)_$($Market)" }
Write-Host "Progress:  Get-Content .\$resDir\run.log -Tail 20 -Wait"
Write-Host "Finished when .\$resDir\DONE exists. Then commit: git add $resDir; git commit -m 'search results'; git push"