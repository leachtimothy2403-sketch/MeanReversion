<#
MeanReversion — start_narrowed_search.ps1

One-command launcher for the narrowed neighborhood search
(`mean_reversion_narrowed.py`, see that file's own docstring for what's
narrowed and why -- seeded from the 3 real Gate2+Gate3+FTMO survivors
`vps_full_review.py` found among the VPS's 88 static_risk candidates).

Same launch mechanics as `start_search_4x.ps1` (Start-Process calling
py.exe directly, hidden window, redirected log files, BelowNormal
priority, safe across a plain RDP disconnect -- NOT across a log-off or
reboot, see that script's docstring for the full reasoning on why WMI/
cmd.exe was deliberately avoided). This is a separate script rather than
another `start_search_4x.ps1` flag because it targets a different entry
script (`mean_reversion_narrowed.py`, not `mean_reversion.py`) and a
different default asset set (one worker per asset below, not grouped by
bar-count balance -- the narrowed SPACE is cheap enough per-asset that
splitting 1:1 makes more sense than the original's multi-asset groups).

One worker per asset, by default NDX100 + US30 + SPX500 (Tim's explicit
2026-08-27 request: "set up workers on US30 and SPX, as well as nasdaq
(one each)"). Worth flagging plainly: the narrowed SPACE's specific
ranges (consolidation_atr_mult~1.0, static_risk_pct in the bottom half
of the grid, exit_horizon_bars near the grid's max, etc.) were derived
ONLY from NDX100 survivors -- running the same narrowed ranges on
US30/SPX500 is a reasonable diversification bet (same strategy family,
untested assets), not a validated result for those two yet. Needs
`mr_precomputed_US30.parquet` / `mr_precomputed_SPX500.parquet` already
present (both were precomputed on the VPS previously as part of the
original 4-way `start_search_4x.ps1` runs -- if either is missing, run
`py -3 precompute.py US30` / `SPX500` first, with `$env:MR_DUKASCOPY_
ROOT` set to the VPS's own Dukascopy cache path, see VPS_DEPLOYMENT.md).

Usage:
    .\start_narrowed_search.ps1                          # NDX100, US30, SPX500 -- one worker each, 50,000-iteration ceiling
    .\start_narrowed_search.ps1 -Iterations 100000
    .\start_narrowed_search.ps1 -Assets NDX100,US30,SPX500,GER40   # add/remove assets
    .\start_narrowed_search.ps1 -OutputSuffix v2                   # fresh campaign, doesn't touch existing output dirs

Every run auto-resumes from its own checkpoint.json if one exists --
this script doubles as the restart command after a crash or a stop, no
separate "resume" step needed. Use a new -OutputSuffix any time
`mean_reversion_narrowed.py`'s SPACE overrides change, so a resumed run
never mixes candidates scored under two different narrowed spaces in
the same top_strategies.json (same convention start_search_4x.ps1's own
docstring documents for the un-narrowed search).

Stop everything:
    Get-Process -Name py -ErrorAction SilentlyContinue | Stop-Process
    (hard-kills every py process on the machine, including any OTHER
    search running alongside this one -- loses up to the last
    checkpoint, i.e. up to 500 iterations / 30 minutes of progress per
    run. Ctrl+C in a specific window is the clean single-run stop, but
    this script hides its windows by default -- use Stop-Process on
    that one PID instead, printed when this script launches it, if you
    only want to stop one worker.)
#>

param(
    [int]$Iterations = 50000,
    [string[]]$Assets = @("NDX100", "US30", "SPX500"),
    [string]$OutputSuffix = "narrowed"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$suffixTag = if ($OutputSuffix) { "_$OutputSuffix" } else { "" }
Write-Host "Launching $($Assets.Count) narrowed-search worker(s), one per asset ($($Assets -join ', ')), $Iterations iterations each (ceiling -- stop manually with Ctrl+C or Stop-Process once your time budget is up)."

foreach ($asset in $Assets) {
    $env:MR_ASSET_FILTER = $asset
    $env:MR_OUTPUT_DIR   = "meanreversion_output$suffixTag`_$asset"
    $env:MR_ITERATIONS   = "$Iterations"

    $log    = Join-Path $root "narrowed$suffixTag`_$asset`_console.log"
    $errlog = Join-Path $root "narrowed$suffixTag`_$asset`_console.err.log"

    Start-Process -FilePath "py" -ArgumentList @("-3", "mean_reversion_narrowed.py") `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError $errlog

    Write-Host "  [$asset] -> $env:MR_OUTPUT_DIR ($log)"
}

Start-Sleep -Seconds 5
$procs = Get-Process -Name py -ErrorAction SilentlyContinue
$procs | ForEach-Object { $_.PriorityClass = 'BelowNormal' }
Write-Host "`n$($procs.Count) py process(es) running at BelowNormal priority (this includes any OTHER search already running on the machine, not just these workers)."
Write-Host "Tail progress with, e.g.:  Get-Content .\narrowed$suffixTag`_NDX100_console.log -Wait -Tail 5"
Write-Host "Push progress back periodically so it can be reviewed remotely, e.g.:"
Write-Host "  git add meanreversion_output$suffixTag`_*/top_strategies.json; git commit -m 'Narrowed search progress'; git push"
Write-Host "Safe to disconnect the RDP session now (don't log off) -- these keep running."
