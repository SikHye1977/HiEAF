# Always run project work with this workspace's virtual environment.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run scripts/setup_windows.ps1 first.' }
$env:PYTHONUTF8 = '1'
$env:PATH = (Split-Path -Parent $projectPython) + [IO.Path]::PathSeparator + $env:PATH
$env:FFMPEG_EXE = Join-Path (Split-Path -Parent $projectPython) 'ffmpeg.exe'
Push-Location $projectRoot
try {
    & $projectPython @args
    $projectExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $projectExitCode
