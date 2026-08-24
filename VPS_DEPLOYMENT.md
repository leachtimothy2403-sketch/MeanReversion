# Running MeanReversion on the VPS

This repo has no VPS launcher scripts of its own yet — RCTBE's own
`_start_vps_layer2_search.ps1` / `_stop_vps_layer2_search.ps1` (and
siblings) are proven-correct on your actual VPS already (WMI-detached
process creation + Idle OS priority, so a search survives RDP disconnect
and never contends with a live MT5 terminal — see RCTBE's `README.md`,
"General pattern for anything new"). Writing a fresh, untested version
of that same PowerShell risks a subtle `cmd.exe` quoting bug being
trusted for an unattended multi-hour run with nobody watching it — worse
than not having the convenience script at all. The safer path is either
run this in the foreground for now, or copy one of RCTBE's own working
launcher scripts into this repo and repoint it at `mean_reversion.py`
(same working directory pattern, just a different target script and a
different env var prefix — `MR_*` instead of `RCTBE_L2*`) — that's a
much smaller, lower-risk edit than writing one from scratch.

## Minimal path (foreground, RDP session left open)

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

## Detached / survives-disconnect path

Adapt RCTBE's own `_start_vps_layer2_search.ps1` (copy it into this repo,
rename, and change: the working directory to this repo's path, the
target script from `layer2_optimizer.py` to `mean_reversion.py`, and the
env var prefix from `RCTBE_L2_*` to `MR_*` — `RCTBE_L2_ITERATIONS`  ->
`MR_ITERATIONS`, `RCTBE_L2_ASSET_FILTER` -> `MR_ASSET_FILTER`, etc., per
`README.md`'s env var table). Everything else about the pattern —
WMI `Invoke-CimMethod Win32_Process Create`, `start ... /low`, output to
a per-run `*.log` file, checkpoint-every-500-iterations/30-min so a stop
is always safe to resume — carries over unchanged, since
`mean_reversion.py` uses the exact same checkpoint/resume/env-var
conventions as `layer2_optimizer.py` on purpose.

## Sending results back

```powershell
git add mr_precomputed_*.parquet meanreversion_output/top_strategies.json
git commit -m "Mean-reversion search results"
git push
```

(`mr_precomputed_*.parquet` and `meanreversion_output/top_strategies.json`
are the only large/binary artifacts worth versioning — everything else
in `meanreversion_output/` is gitignored as regenerable working state,
see `.gitignore`.)
