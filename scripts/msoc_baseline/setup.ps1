$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Workspace .venv is required.' }
& $python -m pip install --no-deps -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed.' }
$snapshot = Join-Path $PSScriptRoot 'upstream'
if (-not (Test-Path -LiteralPath $snapshot)) {
    $staging = Join-Path $env:TEMP ('msoc-download-' + [guid]::NewGuid().ToString('N'))
    & git clone --no-checkout https://github.com/bok-bok/MSOC.git $staging
    if ($LASTEXITCODE -ne 0) { throw 'Official repository download failed.' }
    $archive = Join-Path $staging 'official.zip'
    & git -C $staging archive --format=zip --output=$archive 0014452f1efd1ca18b38357b0c25f0ff9df87c09 src README.md environment.yml
    if ($LASTEXITCODE -ne 0) { throw 'Pinned official commit unavailable.' }
    Expand-Archive -LiteralPath $archive -DestinationPath $snapshot
}
Push-Location $root
try {
    # Keep embedded Python quotes intact on Windows PowerShell 5.1.
@'
from scripts.msoc_baseline.model import verify_upstream
print("Verified official MSOC:", verify_upstream())
'@ | & $python -
    if ($LASTEXITCODE -ne 0) { throw 'Official source verification failed.' }
} finally { Pop-Location }
