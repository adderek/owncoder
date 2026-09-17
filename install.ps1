<#
.SYNOPSIS
    Installs owncoder on Windows, inside WSL2.

.DESCRIPTION
    owncoder does not run natively on Windows. Its shell and web tools execute
    inside a bubblewrap/seccomp sandbox, and its file locking and resource
    limits use fcntl/resource — all Linux kernel interfaces with no Windows
    equivalent. Rather than ship a build with the sandbox stubbed out (a
    security boundary that silently does nothing is worse than none), this
    script sets up WSL2 and installs there.

    Run:  irm https://raw.githubusercontent.com/adderek/owncoder/master/install.ps1 | iex

.PARAMETER Distro
    WSL distribution to install into. Defaults to the configured default one.

.PARAMETER InstallWsl
    Install WSL2 if it is missing, instead of only reporting it. Needs an
    elevated shell and, usually, a reboot.
#>
[CmdletBinding()]
param(
    [string]$Distro = '',
    [switch]$InstallWsl
)

$ErrorActionPreference = 'Stop'
$InstallUrl = 'https://raw.githubusercontent.com/adderek/owncoder/master/install.sh'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "warning: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "error: $msg" -ForegroundColor Red }

# ── is WSL there at all? ────────────────────────────────────────────────────
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) {
    Write-Fail 'WSL is not installed.'
    Write-Host ''
    Write-Host 'In an Administrator PowerShell, run:'
    Write-Host '    wsl --install'
    Write-Host 'reboot, finish the Linux user setup, then run this script again.'
    exit 1
}

# `wsl --list --quiet` prints nothing (and exits non-zero on some builds) when
# the feature is present but no distribution has been installed.
$distros = @(& wsl.exe --list --quiet 2>$null |
             ForEach-Object { $_ -replace "`0", '' } |
             Where-Object { $_.Trim() -ne '' })

if ($distros.Count -eq 0) {
    if ($InstallWsl) {
        Write-Step 'Installing WSL2 with the default distribution (Ubuntu)...'
        & wsl.exe --install -d Ubuntu
        Write-Host ''
        Write-Host 'Reboot if prompted, finish the Linux username/password setup,'
        Write-Host 'then run this script again to install owncoder.'
        exit 0
    }
    Write-Fail 'WSL is present but has no Linux distribution installed.'
    Write-Host ''
    Write-Host 'Install one:'
    Write-Host '    wsl --install -d Ubuntu'
    Write-Host 'or re-run this script with -InstallWsl from an Administrator shell.'
    exit 1
}

if ($Distro -eq '') {
    $Distro = $distros[0].Trim()
}
Write-Step "Using WSL distribution: $Distro"

# ── prerequisites inside the distro ─────────────────────────────────────────
# -l gives a login shell so PATH matches what the user will see later.
& wsl.exe -d $Distro -e bash -lc 'command -v curl >/dev/null' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Step 'Installing curl and git inside the distribution...'
    & wsl.exe -d $Distro -e bash -lc `
        'sudo apt-get update && sudo apt-get install -y curl git ripgrep bubblewrap'
    if ($LASTEXITCODE -ne 0) {
        Write-Fail 'Could not install prerequisites. Open the Linux shell and install curl and git yourself, then re-run.'
        exit 1
    }
}

# ── install ─────────────────────────────────────────────────────────────────
Write-Step 'Installing owncoder inside WSL...'
& wsl.exe -d $Distro -e bash -lc "curl -LsSf $InstallUrl | sh"
if ($LASTEXITCODE -ne 0) {
    Write-Fail 'The Linux installer failed. Its output is above.'
    exit $LASTEXITCODE
}

Write-Host ''
Write-Step 'Done.'
Write-Host "Open a Linux shell with:  wsl -d $Distro"
Write-Host 'Then:  owncoder setup    (once), and  agent chat  inside a project.'
Write-Host ''
Write-Warn 'Keep projects on the Linux filesystem (~/code/...), not /mnt/c/.'
Write-Host '  Cross-filesystem access is slow enough to dominate indexing time,'
Write-Host '  and Windows file permissions do not survive the translation.'
