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
# Send source over stdin: Windows PowerShell 5.1 strips embedded quotes in -c arguments.
@'
import importlib.util
import sys

spec = importlib.util.find_spec("torch")
version = __import__("torch").__version__ if spec else None
print("Existing torch:", version)
sys.exit(10 if version is not None and version != "2.7.1+cu128" else 0)
'@ | & $python -
$torchCheckExitCode = $LASTEXITCODE
if ($torchCheckExitCode -eq 10) { throw 'Existing Torch differs. Use a separate checkout/.venv; do not modify a HiEAF environment.' }
if ($torchCheckExitCode -ne 0) { throw 'Could not check existing Torch. See the Python error above; no packages have been installed.' }
Write-Host '[SETUP] Updating pip inside the workspace .venv...'
& $python -m pip install --index-url https://pypi.org/simple --only-binary=:all: --upgrade pip==25.3
if ($LASTEXITCODE -ne 0) { throw 'Workspace pip update failed.' }
Write-Host '[SETUP] Installing Torch dependencies from PyPI...'
& $python -m pip install --index-url https://pypi.org/simple --only-binary=:all: -r (Join-Path $PSScriptRoot 'requirements-torch-blackwell.txt')
if ($LASTEXITCODE -ne 0) { throw 'Torch dependency installation from PyPI failed.' }
# Dependencies were resolved above against PyPI. Do not search the CUDA-only
# index for general packages or their build backends (e.g. typing_extensions/flit_core).
Write-Host '[SETUP] Installing official PyTorch 2.7.1/CUDA 12.8 (large download)...'
& $python -m pip install --no-deps --only-binary=:all: torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA 12.8 PyTorch installation failed.' }
Write-Host '[SETUP] Installing MSOC dependencies from PyPI...'
& $python -m pip install --index-url https://pypi.org/simple -r (Join-Path $PSScriptRoot 'requirements-blackwell.txt')
if ($LASTEXITCODE -ne 0) { throw 'MSOC dependencies installation failed.' }
& (Join-Path $PSScriptRoot 'setup.ps1')
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed package dependencies are incomplete or incompatible.' }
Push-Location $root
try {
    & $python -m scripts.msoc_baseline.check_environment
    if ($LASTEXITCODE -ne 0) { throw 'GPU/runtime check failed. Resolve this before training.' }
} finally { Pop-Location }
