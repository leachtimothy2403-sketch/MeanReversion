<#
MeanReversion — start_search_4x.ps1

One-command launcher for a 4-way parallel search on one machine. Splits
the 7 index assets across 4 groups (grouped so each group's average
per-asset bar count is close to the overall average -- see the ASSET
GROUPS comment below -- so all 4 processes should run at roughly the
same iterations/sec), launches each as its own `py -3 mean_reversion.py`
process, and sets them to BelowNormal priority so they don't starve
anything else running on the machine (e.g. a live MT5 terminal).

Deliberately NOT built on RCTBE's own `_start_vps_layer2_search.ps1`
pattern (WMI Win32_Process Create + cmd.exe) -- that's proven-correct on
RCTBE's own VPS already, but re-deriving untested cmd.exe/WMI quoting
for a second project risked exactly the kind of subtle bug that's easy
to miss and expensive to discover 10 hours into an unattended run (see
this repo's git history / VPS_DEPLOYMENT.md for the full reasoning).
This script sidesteps that whole risk class: `Start-Process` here takes
a plain PowerShell array as -ArgumentList, calling py.exe directly with
no intermediate shell to mis-parse a quote or a redirect operator.

What this does NOT do: survive a full LOG OFF (not just a disconnect).
Windows keeps an RDP session's processes running across a plain
disconnect, so `Start-Process ... -WindowStyle Hidden` here is enough
for "walk away and reconnect later." If you need it to survive an
actual log-off or reboot, that needs Task Scheduler (schtasks with
"run whether user is logged on or not") or a Windows service -- ask if
you want that built instead; it's a different (well-trodden, low-risk)
mechanism, not a reason to reach for WMI.

Usage:
    .\start_search_4x.ps1                  # all 4 groups, 100,000-iteration ceiling
    .\start_search_4x.ps1 -Iterations 50000
    .\start_search_4x.ps1 -RunIds 2,3,4    # skip group 1 (e.g. to cut memory load)
    .\start_search_4x.ps1 -OutputSuffix v2 # fresh campaign, doesn't touch existing output dirs
    .\start_search_4x.ps1 -Asset NDX100 -NumWorkers 3 -OutputSuffix ndx100_focus
                                            # -Asset overrides the group table entirely:
                                            # NumWorkers identical processes, all searching
                                            # only that one asset (higher combined throughput
                                            # on it than splitting across the usual groups,
                                            # and each process's memory footprint is just that
                                            # one asset's data -- lighter than any multi-asset
                                            # group above).

-RunIds lets you launch a subset of the groups below -- each launched
process only loads its OWN group's assets into memory (via
MR_ASSET_FILTER), so dropping a group proportionally cuts total RAM use
across the whole machine, not just that one process's share. Re-run with
the full set (or just the dropped group's id) later to pick up the
assets you skipped -- each run resumes independently from its own
checkpoint either way (see below), so partial runs are never wasted.
Ignored when -Asset is set.

Every run auto-resumes from its own checkpoint.json if one exists, so
this script is also the restart command after a crash or a stop -- no
separate "resume" step needed. IMPORTANT: that auto-resume is why
-OutputSuffix exists -- if mean_reversion.py's own SPACE/scoring logic
has changed (a new swept parameter, a changed threshold) since a given
output dir's checkpoint was written, resuming into it mixes candidates
scored under two different rules in the same top_strategies.json/
checkpoint.json. Use a new -OutputSuffix any time you've changed the
strategy code itself, so the new campaign starts clean in its own
meanreversion_output_run<N>_<suffix> dirs and the old results stay
untouched for comparison.

Stop everything:
    Get-Process -Name py -ErrorAction SilentlyContinue | Stop-Process
    (hard-kills each process -- loses up to the last checkpoint, i.e.
    up to 500 iterations / 30 minutes of progress on that run, same as
    pulling the plug. Ctrl+C in a specific window instead is the clean
    way to stop just that one run -- KeyboardInterrupt is caught by
    mean_reversion.py's try/finally, which saves immediately.)
#>

param(
    [int]$Iterations = 100000,
    [int[]]$RunIds = @(1, 2, 3, 4),
    [string]$OutputSuffix = "",
    [string]$Asset = "",
    [int]$NumWorkers = 3
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# ── ASSET GROUPS ─────────────────────────────────────────────────────
# Balanced by average per-asset 1-min bar count within each group (not
# by asset count) since mean_reversion.py samples one asset uniformly
# at random per iteration -- what drives an iteration's cost is the
# SIZE of whichever asset got picked, so groups with similar average
# asset size should see similar iterations/sec, not groups with an
# equal asset COUNT. Based on the IS-only-truncated bar counts observed
# 2026-08 (~1.1-1.7M bars/asset); rebalance if a future precompute run
# shows materially different sizes.
$allGroups = @(
    @{ id = 1; assets = "NDX100,FRA40" }
    @{ id = 2; assets = "SPX500,UK100" }
    @{ id = 3; assets = "US30,GER40" }
    @{ id = 4; assets = "JPN225" }
)
if ($Asset) {
    # Single-asset mode: NumWorkers identical processes all searching
    # only this one asset -- multiplies combined throughput on it
    # instead of splitting across the usual asset groups. Each process
    # loads just this one asset's data, so this is lighter per-process
    # than any of the multi-asset groups above.
    $groups = 1..$NumWorkers | ForEach-Object { @{ id = $_; assets = $Asset } }
} else {
    $groups = $allGroups | Where-Object { $RunIds -contains $_.id }
}

if ($groups.Count -eq 0) {
    Write-Error "No groups matched -RunIds $($RunIds -join ','). Valid ids are 1-4."
    exit 1
}

$suffixTag = if ($OutputSuffix) { "_$OutputSuffix" } else { "" }
if ($Asset) {
    Write-Host "Launching $($groups.Count) parallel searches, all on $Asset only, $Iterations iterations each (ceiling -- stop manually with Ctrl+C or Stop-Process once your time budget is up)."
} else {
    Write-Host "Launching $($groups.Count) of 4 parallel searches (ids: $($groups.id -join ',')), $Iterations iterations each (ceiling -- stop manually with Ctrl+C or Stop-Process once your time budget is up)."
}
if ($OutputSuffix) { Write-Host "Output suffix '$OutputSuffix' -- fresh campaign, existing output dirs untouched." }

foreach ($g in $groups) {
    $env:MR_ASSET_FILTER = $g.assets
    $env:MR_OUTPUT_DIR   = "meanreversion_output_run$($g.id)$suffixTag"
    $env:MR_ITERATIONS   = "$Iterations"

    $log    = Join-Path $root "run$($g.id)$suffixTag`_console.log"
    $errlog = Join-Path $root "run$($g.id)$suffixTag`_console.err.log"

    Start-Process -FilePath "py" -ArgumentList @("-3", "mean_reversion.py") `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError $errlog

    Write-Host "  [run$($g.id)$suffixTag] assets=$($g.assets) -> $log"
}

Start-Sleep -Seconds 5
$procs = Get-Process -Name py -ErrorAction SilentlyContinue
$procs | ForEach-Object { $_.PriorityClass = 'BelowNormal' }
Write-Host "`n$($procs.Count) py process(es) running at BelowNormal priority."
Write-Host "Tail progress with, e.g.:  Get-Content .\run1_console.log -Wait -Tail 5"
Write-Host "Safe to disconnect the RDP session now (don't log off) -- these keep running."
