# Replay the three pre-registered wider TP/SL variants (threshold 0.65, Jun 1 - Nov 3). Read-only; ~20 min each.
$ErrorActionPreference = "Stop"
$Python = "C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

$variants = @(
    @{ Name = "A_tp2.0_sl2.0_hold90";  Tp = 2.0; Sl = 2.0; Hold = 90 },
    @{ Name = "B_tp2.5_sl2.0_hold180"; Tp = 2.5; Sl = 2.0; Hold = 180 },
    @{ Name = "C_tp3.0_sl2.0_hold240"; Tp = 3.0; Sl = 2.0; Hold = 240 }
)
foreach ($v in $variants) {
    $out = "forward_reports/20261104_exit_variant_$($v.Name)"
    Write-Host "=== $($v.Name) -> $out"
    & $Python -B shadow_replay_v1.py --start 2026-06-01 --end 2026-11-03 --threshold 0.65 `
        --tp-atr $v.Tp --sl-atr $v.Sl --max-hold $v.Hold --output $out
    if ($LASTEXITCODE -ne 0) { throw "$($v.Name) failed with exit code $LASTEXITCODE" }
}
