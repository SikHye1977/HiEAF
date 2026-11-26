$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Workspace .venv is required.' }
& $python -m pip install --index-url https://pypi.org/simple --no-deps -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed.' }
$snapshot = Join-Path $PSScriptRoot 'upstream'
$sourceLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'upstream.lock.json') -Raw | ConvertFrom-Json

function Test-MsocSnapshot([string]$Directory) {
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { return $false }
    foreach ($entry in $sourceLock.files.PSObject.Properties) {
        $file = Join-Path $Directory $entry.Name
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            Write-Host "[SETUP] Missing official source: $($entry.Name)"
            return $false
        }
        $hasher = [Security.Cryptography.SHA256]::Create()
        try {
            $actual = [BitConverter]::ToString($hasher.ComputeHash([IO.File]::ReadAllBytes($file))).Replace('-', '').ToLowerInvariant()
        } finally { $hasher.Dispose() }
        if ($actual -ne $entry.Value) {
            Write-Host "[SETUP] Official source checksum differs: $($entry.Name)"
            return $false
        }
    }
    return $true
}

if (-not (Test-MsocSnapshot $snapshot)) {
    $staging = Join-Path $root ('reports/msoc_setup/source-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    $repository = Join-Path $staging 'repository'
    # git archive can apply Windows core.autocrlf even without a checkout.
    # Override it per command; do not change the user's global Git settings.
    & git -c core.autocrlf=false -c core.eol=lf clone --no-checkout https://github.com/bok-bok/MSOC.git $repository
    if ($LASTEXITCODE -ne 0) { throw 'Official repository download failed.' }
    $archive = Join-Path $staging 'official.zip'
    & git -C $repository -c core.autocrlf=false -c core.eol=lf archive --format=zip --output=$archive $sourceLock.commit src README.md environment.yml
    if ($LASTEXITCODE -ne 0) { throw 'Pinned official commit unavailable.' }
    $prepared = Join-Path $staging 'verified_upstream'
    Expand-Archive -LiteralPath $archive -DestinationPath $prepared
    if (-not (Test-MsocSnapshot $prepared)) {
        throw 'Fresh official source failed exact checksum verification. Existing upstream has not been changed.'
    }
    $backup = Join-Path $staging 'previous_upstream'
    # Check all absolute move targets before moving any directories.
    foreach ($candidate in @($snapshot, $prepared, $backup)) {
        $absolute = [IO.Path]::GetFullPath($candidate)
        if (-not $absolute.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Source installation path is outside this checkout: $absolute"
        }
        if ((Test-Path -LiteralPath $candidate) -and ((Get-Item -LiteralPath $candidate).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Source installation path must not be a junction or symbolic link: $absolute"
        }
    }
    if (Test-Path -LiteralPath $snapshot) {
        Move-Item -LiteralPath $snapshot -Destination $backup
        Write-Host "[SETUP] Preserved previous upstream: $backup"
    }
    Move-Item -LiteralPath $prepared -Destination $snapshot
}
Write-Host "[SETUP] Exact official source checksums passed ($($sourceLock.files.PSObject.Properties.Name.Count) files)."
Push-Location $root
try {
    # Keep embedded Python quotes intact on Windows PowerShell 5.1.
@'
from scripts.msoc_baseline.model import verify_upstream
print("Verified official MSOC:", verify_upstream())
'@ | & $python -
    if ($LASTEXITCODE -ne 0) { throw 'Official source verification failed.' }
} finally { Pop-Location }
