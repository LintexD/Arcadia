# Launch Arcadia Voxel Engine (Vulkan 1.3 / RTX 3080)
# Note: The dev profile in Cargo.toml is already configured with `opt-level = 2`
# and pre-compiled in `target/debug`, so `cargo run` launches immediately at high speed!
param(
    [switch]$Release,
    [switch]$Benchmark,
    [switch]$AutoExit,
    [int]$Frames = 0,
    [string]$Scene = "",
    [string]$TimeOfDay = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
}

$extraArgs = @()
if ($Benchmark) { $extraArgs += "--bench" }
if ($AutoExit) { $extraArgs += "--auto-exit" }
if ($Frames -gt 0) { $extraArgs += "--frames", $Frames.ToString() }
if ($Scene -ne "") { $extraArgs += "--scene", $Scene }
if ($TimeOfDay -ne "") { $extraArgs += "--tod", $TimeOfDay }

if ($Release) {
    Write-Host "Building & launching Arcadia Voxel Engine (--release, LTO)..." -ForegroundColor Cyan
    if ($extraArgs.Count -gt 0) {
        cargo run --release -j 2 -p arcadia-app -- $extraArgs
    } else {
        cargo run --release -j 2 -p arcadia-app
    }
} else {
    Write-Host "Launching Arcadia Voxel Engine (opt-level = 2)..." -ForegroundColor Cyan
    if ($extraArgs.Count -gt 0) {
        cargo run -j 2 -p arcadia-app -- $extraArgs
    } else {
        cargo run -j 2 -p arcadia-app
    }
}
