# open_exit_search launcher (VPS). Handoff steps 3 + 5: exit management of the opening trade, and the opening
# trade + FVG legs transplanted to other opens (NQ, NDX100, US30, SPX500 at 09:30 NY; GER40, FRA40 09:00; UK100 08:00).
# Runs prep -> stage0 (ablation) -> SHORT track s1/s2/s3 -> LONG track s1/s2/s3 -> stage4 (FTMO 2-step combos) ->
# stage5 (FundedNext Flex for NQ) in ONE hidden, BelowNormal-priority Python process with a worker pool.
# Resumable: re-running skips stage-1 work already saved and any later stage whose output file exists.
#
#   cd C:\Users\Administrator\MeanReversion\open_exit_search
#   .\start_open_exit_search.ps1                          # defaults: 4 workers, 50k short + 35k long configs per instrument (~8 h)
#   .\start_open_exit_search.ps1 -Workers 8               # if the VPS has more cores ($env:NUMBER_OF_PROCESSORS)
#   .\start_open_exit_search.ps1 -N1 30000 -N1Long 20000  # shorter run (~5 h on 4 workers)
#
# Progress:   Get-Content .\results\run.log -Tail 30 -Wait
# Finished:   .\results\DONE exists.  Then: git add results; git commit -m "open_exit_search results"; git push
# Stop:       Stop-Process -Id (Get-Content .\logs\pid.txt); re-run this script to resume
param(
    [int]$Workers = 4,
    [int]$N1 = 50000,
    [int]$N1Long = 35000,
    [int]$Top2 = 150,
    [int]$Top2Long = 60,
    [int]$NN = 16,
    [int]$NNLong = 12,
    [int]$K3 = 15,
    [int]$K4 = 3,
    [int]$Top4 = 80,
    [string]$Inst = "NQ,NDX100,US30,SPX500,GER40,FRA40,UK100"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$need = @{ "NQ" = "..\NASDAQFuturesData\nq_continuous_1min.parquet"; "NDX100" = "..\CFDData\ndx100_dukascopy_1min.parquet";
           "US30" = "..\CFDData\us30_dukascopy_1min.parquet"; "SPX500" = "..\CFDData\spx500_dukascopy_1min.parquet";
           "GER40" = "..\CFDData\ger40_dukascopy_1min.parquet"; "FRA40" = "..\CFDData\fra40_dukascopy_1min.parquet";
           "UK100" = "..\CFDData\uk100_dukascopy_1min.parquet" }
$missing = @()
foreach ($k in $Inst.Split(",")) { if (-not (Test-Path (Join-Path $here $need[$k]))) { $missing += $need[$k] } }
if ($missing.Count -gt 0) {
    Write-Host "Missing data files (copy them from the laptop, same relative paths; data is not in git):" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
Write-Host "Using: $(py -3 --version 2>&1)"
Write-Host "Checking Python packages (numpy, pandas, pyarrow, tzdata)..."
py -3 -m pip install --quiet numpy pandas pyarrow tzdata
New-Item -ItemType Directory -Force -Path (Join-Path $here "logs") | Out-Null
$argLine = "-3 oe_search.py all --workers $Workers --n1 $N1 --n1-long $N1Long --top2 $Top2 --top2-long $Top2Long --nn $NN --nn-long $NNLong --k3 $K3 --k4 $K4 --top4 $Top4 --inst $Inst"
$p = Start-Process -FilePath "py" -ArgumentList $argLine -WorkingDirectory $here -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $here "logs\stdout.log") -RedirectStandardError (Join-Path $here "logs\stderr.log")
Start-Sleep -Seconds 2
try { $p.PriorityClass = "BelowNormal" } catch {}
Set-Content -Path (Join-Path $here "logs\pid.txt") -Value $p.Id
Write-Host "Started open_exit_search (PID $($p.Id)), $Workers workers, N1=$N1 short / $N1Long long per instrument." -ForegroundColor Green
Write-Host "The log prints an ETA for each stage-1 run; the long-track stage 1 is the longest part."
Write-Host "Progress:  Get-Content .\results\run.log -Tail 30 -Wait"
Write-Host "Errors:    Get-Content .\logs\stderr.log -Tail 30"
Write-Host "Finished when .\results\DONE exists. Then: git add results; git commit -m 'open_exit_search results'; git push"
