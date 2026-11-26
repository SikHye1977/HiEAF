# Blackwell Windows workstation: MSOC comparison

This adapter uses the official [bok-bok/MSOC](https://github.com/bok-bok/MSOC)
commit `0014452f1efd1ca18b38357b0c25f0ff9df87c09` on the existing HiEAF splits.
It does not claim to reproduce the paper's original dataset splits or numbers.
`upstream.lock.json` verifies all 64 official snapshot files. Setup downloads
the original files without editing them; Git contains the adapter and checksums.
See README for preprocessing, sampling and implementation differences.

## 1. Pull and install (PowerShell, repository root)

Use a dedicated checkout with Python 3.10 and Git installed. Keep any existing
HiEAF environment intact; never copy the old CUDA 12.1 `.venv` to Blackwell.

```powershell
git pull --ff-only origin main
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
./scripts/msoc_baseline/setup_blackwell.ps1
```

Setup creates the checkout's `.venv`, installs **PyTorch 2.7.1/CUDA 12.8** and
pinned MSOC dependencies, downloads pinned official sources, and checks CUDA/BF16
kernels. It refuses to replace a different existing Torch build. It does not
install HiEAF models or run a dataset experiment. A compatible NVIDIA driver
is required. Blackwell hardware cannot be tested from the source PC; the
destination GPU check and preflight must pass before training.

The installer updates pip to 25.3 in this checkout only. General dependencies
come from PyPI; `requirements-torch-blackwell.txt` lists the official Windows
Python 3.10 Torch wheel's runtime requirements. Only the pinned CUDA Torch wheel
is installed from the PyTorch index, with `--no-deps` after those requirements
have been resolved. This avoids trying to build general packages using the
CUDA index alone. `pip check` verifies the complete installed environment before
the GPU check. After an interrupted dependency download, rerun the same setup;
there is no need to delete the `.venv` or pip cache.

Official installation references:
- https://pytorch.org/get-started/previous-versions/#v271
- https://pytorch.org/blog/pytorch-2-7/

## 2. Frozen metadata and paths

Video files alone are insufficient. Copy the supplied `MSOC_metadata_20261126.zip`
and checksum to Blackwell and extract into a NEW directory. Example paths:

```powershell
Get-FileHash 'E:/incoming/MSOC_metadata_20261126.zip' -Algorithm SHA256
# Compare with the supplied .zip.sha256 before extraction.
Expand-Archive -LiteralPath 'E:/incoming/MSOC_metadata_20261126.zip' -DestinationPath 'E:/MSOC_metadata'
Copy-Item './scripts/msoc_baseline/data.example.json' './scripts/msoc_baseline/data.local.json'
notepad './scripts/msoc_baseline/data.local.json'
```

Edit the four roots in `data.local.json` to match this workstation. Examples
assume `E:/datasets/FakeAVCeleb_v1.2`, `E:/datasets/AV-Deepfake1M`, and
`E:/datasets/PolyGlotFake`. Point to the inner dataset folder if extraction added
another directory level. Keep original relative file paths. The metadata root
contains `configs/`, `reports/`, `scripts/`, `datasets/`. No frozen IDs, splits,
labels or null labels are rewritten.

The previous AVDF1M-only ZIP is NOT the complete MSOC metadata bundle. This new
bundle includes FAV/PGF manifests, PGF JSON labels and the portable HiEAF reference.
Old absolute Windows result directories, HiEAF weights and feature caches are
not needed. The sealed reference preserves 54 rows and verified evidence hashes;
new test IDs/clip labels must match the exported per-target digests exactly.

Expected train/dev/test counts:
- FAV: 14724 / 3147 / 3126
- AVDF1M: 14724 / 3147 / 120327
- PGF: 9948 / 2092 / 2082

To recreate the metadata ZIP on the evidence-holding PC, use its project venv:
`python -m scripts.msoc_baseline.package_metadata --output <new-archive.zip>`.
`export_reference` is only for the initial export from preserved original evidence.

## 3. Verify and run

```powershell
$python = './.venv/Scripts/python.exe'
$exp = './scripts/msoc_baseline/experiment.py'
$data = './scripts/msoc_baseline/data.local.json'
$out = './reports/msoc_blackwell_resnet_seed42_v1'
$trainCache = 'E:/msoc_cache/train'
$evalCache = 'E:/msoc_cache/eval'

& $python -m unittest scripts.msoc_baseline.test_pipeline scripts.msoc_baseline.test_portability -v
if ($LASTEXITCODE -ne 0) { throw 'Regression checks failed.' }
& $python $exp preflight --data-config $data --output $out --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; training has not started.' }
& $python $exp run --data-config $data --output $out --device cuda:0 --train-cache $trainCache --eval-cache $evalCache
```

Default: **ResNet, seed42, 30 epochs, batch64, Adam 2e-4, BF16 training/FP32
evaluation**, train three source models and evaluate all nine source/target pairs.
Do not enlarge batch just to fill 96GB VRAM: this changes optimizer and BN conditions.
Workers may be changed without changing the protocol. Input is a single prefix
window of 30 frames (1.2s at 25fps), not full-video MSOC. SCNet is available using
`--backbone scnet` with a different output directory. Additional seeds use `--seed`
and distinct outputs. Input caches can be shared. The paper's multi-seed reporting
is separate from the default single-seed run.

Optional bounded timing before the full run:
`& $python $exp benchmark --data-config $data --output $out --device cuda:0`.
It times synthetic training and a few source-train decodes, not target accuracy.

## 4. Stop, resume, status and results

Press Ctrl+C once and wait for a safe save. Repeat the same `run` command to
resume. Checkpoints preserve model/BN/optimizer/scheduler/scaler/RNG/cursor;
cache samples and prediction batches commit separately. Keep code, environment,
batch, seed, precision and input paths/stamps stable within a run. Use a new
output for changed settings rather than deleting past results.

```powershell
& $python $exp status --output $out
& $python $exp compare --data-config $data --output $out
```

Terminal output includes stage, epoch, counts, throughput, elapsed and stage ETA.
This ETA is not the whole remaining multi-source pipeline. `SUMMARY.csv` contains
MSOC results; `COMPARISON_WITH_HIEAF.csv` contains 54 preserved reference rows plus
9 MSOC rows when all pairs complete. HiEAF A/B refers to its X3D BN interventions;
MSOC uses its own native stored BN. Save exact commands, run artifacts, pip freeze,
hardware and driver details with the paper's results.
