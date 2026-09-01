param(
    [string]$XmTerminalPath = $env:XM_TERMINAL_PATH,
    [int]$InitialDelaySeconds = 0,
    [int]$XmWarmupSeconds = 45
)

$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$GeminiScript = Join-Path $BaseDir "gemini.py"
$CodexScript = Join-Path $BaseDir "codex.py"
$PythonExe = "C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe"
$LogDir = Join-Path $BaseDir "runtime_logs"
$LauncherLog = Join-Path $LogDir "gemini_launcher.log"
$StdoutLog = Join-Path $LogDir "gemini_stdout.log"
$StderrLog = Join-Path $LogDir "gemini_stderr.log"
$CodexStdoutLog = Join-Path $LogDir "codex_stdout.log"
$CodexStderrLog = Join-Path $LogDir "codex_stderr.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-LauncherLog {
    param([string]$Message)

    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $LauncherLog -Value $line -Encoding UTF8
}

if ($InitialDelaySeconds -gt 0) {
    Write-LauncherLog "Waiting initial delay: ${InitialDelaySeconds}s"
    Start-Sleep -Seconds $InitialDelaySeconds
}

function Get-ProcessSnapshot {
    try {
        return Get-CimInstance Win32_Process -ErrorAction Stop
    }
    catch {
        return Get-WmiObject Win32_Process
    }
}

function Test-ProcessCommandLineContains {
    param(
        $Process,
        [string]$Needle
    )

    if ([string]::IsNullOrWhiteSpace($Process.CommandLine)) {
        return $false
    }

    return $Process.CommandLine.IndexOf(
        $Needle,
        [StringComparison]::OrdinalIgnoreCase
    ) -ge 0
}

if (-not (Test-Path -LiteralPath $GeminiScript)) {
    throw "Missing gemini.py beside this launcher: $GeminiScript"
}
if (-not (Test-Path -LiteralPath $CodexScript)) {
    throw "Missing codex.py beside this launcher: $CodexScript"
}

if ([string]::IsNullOrWhiteSpace($XmTerminalPath)) {
    throw "XmTerminalPath is required. Pass -XmTerminalPath or set XM_TERMINAL_PATH."
}

$XmTerminalPath = (Resolve-Path -LiteralPath $XmTerminalPath).ProviderPath
$XmWorkingDir = Split-Path -Parent $XmTerminalPath
$processes = @(Get-ProcessSnapshot)

$xmProcess = $processes | Where-Object {
    $_.ExecutablePath -and
    $_.ExecutablePath.Equals($XmTerminalPath, [StringComparison]::OrdinalIgnoreCase)
} | Select-Object -First 1

if ($xmProcess) {
    Write-LauncherLog "XM already running: pid=$($xmProcess.ProcessId) path=$XmTerminalPath"
}
else {
    Write-LauncherLog "Starting XM: $XmTerminalPath"
    Start-Process -FilePath $XmTerminalPath -WorkingDirectory $XmWorkingDir -WindowStyle Minimized
    Start-Sleep -Seconds $XmWarmupSeconds
}

if (Test-Path -LiteralPath $PythonExe) {
    $pythonPrefix = @()
}
else {
    $pyLauncher = (Get-Command py.exe -ErrorAction SilentlyContinue).Source
    if (-not $pyLauncher) {
        throw "Python 3.10 was not found at $PythonExe, and py.exe is not available."
    }

    $PythonExe = $pyLauncher
    $pythonPrefix = @("-3.10")
}

$GeminiFullPath = (Resolve-Path -LiteralPath $GeminiScript).ProviderPath
$CodexFullPath = (Resolve-Path -LiteralPath $CodexScript).ProviderPath
$processes = @(Get-ProcessSnapshot)
$pythonProcessNames = @("python.exe", "pythonw.exe", "py.exe")
$geminiProcess = $processes | Where-Object {
    $_.Name -in $pythonProcessNames -and
    (Test-ProcessCommandLineContains -Process $_ -Needle $GeminiFullPath)
} | Select-Object -First 1

if ($geminiProcess) {
    Write-LauncherLog "gemini.py already running: pid=$($geminiProcess.ProcessId)"
}
else {
    $pythonArgs = $pythonPrefix + @("-u", $GeminiFullPath)
    Write-LauncherLog "Starting gemini.py with $PythonExe"
    $started = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $pythonArgs `
        -WorkingDirectory $BaseDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru
    Write-LauncherLog "gemini.py started: pid=$($started.Id) stdout=$StdoutLog stderr=$StderrLog"
}

$processes = @(Get-ProcessSnapshot)
$codexProcess = $processes | Where-Object {
    $_.Name -in $pythonProcessNames -and
    (Test-ProcessCommandLineContains -Process $_ -Needle $CodexFullPath)
} | Select-Object -First 1

if ($codexProcess) {
    Write-LauncherLog "codex.py already running: pid=$($codexProcess.ProcessId)"
}
else {
    $codexArgs = $pythonPrefix + @("-u", $CodexFullPath, "--paper")
    Write-LauncherLog "Starting codex.py paper monitor with $PythonExe"
    $codexStarted = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $codexArgs `
        -WorkingDirectory $BaseDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $CodexStdoutLog `
        -RedirectStandardError $CodexStderrLog `
        -PassThru
    Write-LauncherLog "codex.py started: pid=$($codexStarted.Id) stdout=$CodexStdoutLog stderr=$CodexStderrLog"
}
