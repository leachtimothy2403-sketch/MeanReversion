# Running MeanReversion on the VPS

## One-command, 4-way parallel search (`start_search_4x.ps1`)

```powershell
.\start_search_4x.ps1                  # 100,000-iteration ceiling per process (default)
.\start_search_4x.ps1 -Iterations 50000
```

Splits the 7 index assets across 4 groups (balanced by average per-asset
bar count, not asset count — see the script's own comments), launches
one `py -3 mean_reversion.py` process per group, sets all of them to
`BelowNormal` priority, and logs each to its own `run<N>_console.log` in
this directory. Safe to disconnect the RDP session afterward (not log
off — a plain disconnect doesn't kill the processes, only a log-off or
reboot does).

Deliberately built differently from RCTBE's own `_start_vps_layer2_
search.ps1`: that one goes through WMI `Invoke-CimMethod Win32_Process
Create` + `cmd.exe`, proven-correct on RCTBE's VPS but not something
worth re-deriving untested for a second project (a subtle `cmd.exe`
quoting bug is exactly the kind of thing that's invisible until hour 10
of an unattended run). `start_search_4x.ps1` instead calls
`Start-Process -FilePath py -ArgumentList @(...)` directly — no
intermediate shell to mis-parse a quote or a redirect operator, so it
avoids that whole risk class rather than needing to get it right.

To stop everything at once (hard kill — loses up to the last checkpoint,
≤500 iterations / 30 min of progress per run, same as pulling the plug):

```powershell
Get-Process -Name py -ErrorAction SilentlyContinue | Stop-Process
```

To stop just one run cleanly (saves immediately on interrupt): bring
that process's window to the foreground — it's hidden by default, so
instead find and stop it individually via `Get-Process`/`Stop-Process`
on its specific PID (printed when the script launches it), or just use
the hard-kill-all above if losing the last partial checkpoint on that
one run doesn't matter.

This does NOT survive a full log-off or reboot — only a disconnect. If
you need that, Task Scheduler (`schtasks` with "run whether user is
logged on or not") is the standard, low-risk way to get it — ask if you
want that built; it's a different, well-trodden mechanism from the WMI
approach above, not a reason to reach for it.

## Minimal path (foreground, single process, RDP session left open)

```powershell
git clone https://github.com/leachtimothy2403-sketch/MeanReversion.git
cd MeanReversion
py -3 -m pip install --quiet numpy pandas pyarrow

# Point at the VPS's Dukascopy cache — same env var RCTBE's own
# layer1_features.py reads, same "find it or download fresh" choice
# RCTBE's README.md documents:
#   Get-ChildItem -Path C:\Users -Recurse -Directory -Filter "USATECHIDXUSD" -ErrorAction SilentlyContinue | Select-Object FullName
$env:MR_DUKASCOPY_ROOT = "C:\path\to\existing\data\dukascopy"

py -3 precompute.py ALL
py -3 selftest.py                          # optional, cheap plumbing check
$env:MR_ITERATIONS = "200000"
py -3 mean_reversion.py                    # this blocks the console until done
```

## If you need survival across a full log-off/reboot, not just disconnect

`start_search_4x.ps1` above covers "walk away and reconnect later" —
that's all a plain RDP disconnect needs. If the VPS might actually log
you off or reboot mid-search, that needs a different mechanism: Task
Scheduler (`schtasks` with "run whether user is logged on or not") is
the standard, low-risk way to get that — ask if you want it built.
Adapting RCTBE's own `_start_vps_layer2_search.ps1` (WMI `Invoke-
CimMethod Win32_Process Create` + `cmd.exe`) is also still an option —
it's proven-correct on RCTBE's own VPS — but `start_search_4x.ps1`
already covers the common case without that added complexity.

## Sending results back

```powershell
git add mr_precomputed_*.parquet meanreversion_output_run*/top_strategies.json
git commit -m "Mean-reversion search results"
git push
```

(`mr_precomputed_*.parquet` and each run's `top_strategies.json` are the
only large/binary artifacts worth versioning — everything else in
`meanreversion_output_run*/` is gitignored as regenerable working state,
see `.gitignore`. If you ran the single-process minimal path instead,
the output directory is just `meanreversion_output/`.)
