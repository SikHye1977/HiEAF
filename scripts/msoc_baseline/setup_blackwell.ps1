# Run only in a separate Blackwell checkout, not the existing HiEAF environment.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & py -3.10 -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.10 (Windows launcher) first.' }
}
# Refuse to replace an existing experiment's Torch installation.
& $python -c 'import importlib.util, sys; s=importlib.util.find_spec("torch"); v=__import__("torch").__version__ if s else None; print("Existing torch:", v); sys.exit(0 if v is None or v == "2.7.1+cu128" else 1)'
if ($LASTEXITCODE -ne 0) { throw 'Existing Torch differs. Use a separate checkout/.venv; do not modify a HiEAF environment.' }
& $python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA 12.8 PyTorch installation failed.' }
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-blackwell.txt')
if ($LASTEXITCODE -ne 0) { throw 'MSOC dependencies installation failed.' }
& (Join-Path $PSScriptRoot 'setup.ps1')
Push-Location $root
try {
    & $python -m scripts.msoc_baseline.check_environment
    if ($LASTEXITCODE -ne 0) { throw 'GPU/runtime check failed. Resolve this before training.' }
} finally { Pop-Location }
