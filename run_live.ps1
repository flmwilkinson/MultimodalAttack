<#
.SYNOPSIS
  One-command live run of the adaptive orchestrator against local Ollama.

.DESCRIPTION
  Runs preflight (Ollama + models + a live smoke call), an optional cheap
  text-only sanity attempt, then the per-objective optimize controller.
  Stops early with clear guidance if preflight fails.

.EXAMPLE
  .\run_live.ps1 -Objective "your authorized test objective"

.EXAMPLE
  .\run_live.ps1 -Objective "..." -Attempts 24 -Pull        # pull role models first

.NOTES
  Run from the COMET-Local-Ollama-English folder. Authorized security testing only.
#>
param(
  [Parameter(Mandatory = $true)] [string] $Objective,
  [string] $Target  = "qwen2.5vl:7b",
  [string] $Judge   = "qwen2.5vl:7b",
  [string] $Refiner = "qwen2.5vl:7b",
  [int]    $Attempts = 24,
  [int]    $Chunk    = 3,
  [switch] $Pull,          # pull the target/judge/refiner models before running
  [switch] $SkipSanity,    # skip the cheap text-only sanity attempt
  [switch] $TextOnly       # run the optimizer in text-only mode (no image pipeline)
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "`n=== Adaptive live run ===" -ForegroundColor Cyan
Write-Host ("Objective : {0}" -f $Objective)
Write-Host ("Target    : {0}" -f $Target)
Write-Host ("Judge     : {0}" -f $Judge)
Write-Host ("Refiner   : {0}" -f $Refiner)
Write-Host ("Attempts  : {0}  Chunk: {1}" -f $Attempts, $Chunk)

if ($Pull) {
  Write-Host "`n[1/4] Pulling role models ..." -ForegroundColor Cyan
  foreach ($m in @($Target, $Judge, $Refiner | Select-Object -Unique)) {
    Write-Host ("  ollama pull {0}" -f $m)
    ollama pull $m
  }
} else {
  Write-Host "`n[1/4] Skipping pulls (pass -Pull to auto-pull role models)."
}

Write-Host "`n[2/4] Preflight ..." -ForegroundColor Cyan
python preflight.py --target $Target --judge $Judge --refiner $Refiner --smoke
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nPreflight failed. Fix the issues above (or re-run with -Pull), then retry." -ForegroundColor Yellow
  exit 1
}

if (-not $SkipSanity) {
  Write-Host "`n[3/4] Cheap sanity attempt (text-only, 2 attempts) ..." -ForegroundColor Cyan
  python run_adaptive.py pair --objective $Objective --target $Target --judge $Judge --refiner $Refiner --attempts 2 --text-only -v
  if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSanity run failed -- stopping before the full run. See output above." -ForegroundColor Yellow
    exit 1
  }
} else {
  Write-Host "`n[3/4] Skipping sanity run (-SkipSanity)."
}

Write-Host "`n[4/4] Per-objective optimize (full portfolio) ..." -ForegroundColor Cyan
$optArgs = @(
  "run_adaptive.py", "optimize",
  "--objective", $Objective,
  "--target", $Target, "--judge", $Judge, "--refiner", $Refiner,
  "--attempts", $Attempts, "--chunk", $Chunk, "--seed", "1", "-v"
)
if ($TextOnly) { $optArgs += "--text-only" }
python @optArgs

Write-Host "`nDone. Re-run the same objective to see the per-objective learning compound." -ForegroundColor Green
Write-Host "Build a report:" -ForegroundColor Green
Write-Host ("  python process_utils/robustness_eval.py --targets targets.example.json --transcripts results/adaptive/adaptive_transcript.jsonl --judge-model {0} --output reports/adaptive_report" -f $Judge)
