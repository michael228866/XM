param(
    [string]$XmTerminalPath,
    [string]$TaskName = "XM Gemini AutoStart",
    [int]$DelaySeconds = 60
)

$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartScript = Join-Path $BaseDir "start_gemini_background.ps1"

function Get-XmTerminalCandidates {
    $paths = New-Object System.Collections.Generic.List[string]
    $startMenuRoots = @(
        "$env:ProgramData\Microsoft\Windows\Start Menu\Programs",
        "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
    )

    $shell = New-Object -ComObject WScript.Shell
    foreach ($root in $startMenuRoots) {
        if (-not (Test-Path -LiteralPath $root)) {
            continue
        }

        Get-ChildItem -LiteralPath $root -Recurse -Filter *.lnk -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "XM|MetaTrader|MT5" -or $_.DirectoryName -match "XM|MetaTrader|MT5" } |
            ForEach-Object {
                $shortcut = $shell.CreateShortcut($_.FullName)
                if ($shortcut.TargetPath -and (Split-Path -Leaf $shortcut.TargetPath) -ieq "terminal64.exe") {
                    $paths.Add($shortcut.TargetPath)
                }
            }
    }

    @(
        $paths |
            Where-Object { Test-Path -LiteralPath $_ } |
            Sort-Object -Unique
    )
}

if (-not (Test-Path -LiteralPath $StartScript)) {
    throw "Missing launcher script: $StartScript"
}

if ([string]::IsNullOrWhiteSpace($XmTerminalPath)) {
    $candidates = @(Get-XmTerminalCandidates)
    if ($candidates.Count -eq 1) {
        $XmTerminalPath = $candidates[0]
    }
    elseif ($candidates.Count -gt 1) {
        Write-Host "Multiple XM terminals found. Re-run with one explicit -XmTerminalPath:"
        foreach ($candidate in $candidates) {
            Write-Host "  $candidate"
        }
        exit 2
    }
    else {
        throw "No XM terminal64.exe shortcut was found. Pass -XmTerminalPath manually."
    }
}

$XmTerminalPath = (Resolve-Path -LiteralPath $XmTerminalPath).ProviderPath
$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$ActionArgs = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-WindowStyle Hidden",
    "-File `"$StartScript`"",
    "-XmTerminalPath `"$XmTerminalPath`""
) -join " "

$ShortcutArgs = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-WindowStyle Hidden",
    "-File `"$StartScript`"",
    "-XmTerminalPath `"$XmTerminalPath`"",
    "-InitialDelaySeconds $DelaySeconds"
) -join " "

function Install-StartupShortcut {
    $startupDir = [Environment]::GetFolderPath("Startup")
    $shortcutPath = Join-Path $startupDir "$TaskName.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $PowerShellExe
    $shortcut.Arguments = $ShortcutArgs
    $shortcut.WorkingDirectory = $BaseDir
    $shortcut.WindowStyle = 7
    $shortcut.Description = "Start XM Global MT5 and gemini.py after Windows logon."
    $shortcut.Save()

    Write-Host "Registered startup shortcut '$shortcutPath'."
    Write-Host "XM: $XmTerminalPath"
    Write-Host "Launcher: $StartScript"
}

try {
    $action = New-ScheduledTaskAction `
        -Execute $PowerShellExe `
        -Argument $ActionArgs `
        -WorkingDirectory $BaseDir

    $trigger = New-ScheduledTaskTrigger -AtLogOn
    if ($DelaySeconds -gt 0) {
        $trigger.Delay = "PT${DelaySeconds}S"
    }

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Days 0) `
        -MultipleInstances IgnoreNew `
        -StartWhenAvailable `
        -Hidden

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "Start XM Global MT5 and gemini.py after Windows logon." `
        -Force | Out-Null

    Write-Host "Registered task '$TaskName'."
    Write-Host "XM: $XmTerminalPath"
    Write-Host "Launcher: $StartScript"
}
catch [System.UnauthorizedAccessException] {
    Install-StartupShortcut
}
catch {
    $errorText = "$($_.Exception.Message) $($_.FullyQualifiedErrorId)"
    if ($errorText -match "Access is denied|拒絕存取|0x80070005|0x80041003") {
        Install-StartupShortcut
    }
    else {
        throw
    }
}
