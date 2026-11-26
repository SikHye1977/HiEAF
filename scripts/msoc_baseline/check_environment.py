"""Check CUDA kernels on the destination machine, without data or training."""
import json
import sys
from pathlib import Path
import torch
from .common import ROOT, require, runtime_versions


def main():
    require(Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve(), 'Use workspace .venv')
    require(torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver and CUDA-enabled PyTorch')
    device = torch.device('cuda:0')
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 10:
        require(tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8), 'Blackwell requires the CUDA 12.8+ build for this setup')
    require(torch.cuda.is_bf16_supported(), 'BF16 unavailable for the default training configuration')
    # Exercise GEMM, convolution and backward kernels on the actual GPU.
    conv = torch.nn.Conv3d(3, 8, 3, padding=1).to(device)
    x = torch.randn(2, 3, 4, 16, 16, device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = conv(x).float().square().mean()
        y = torch.randn(32, 32, device=device) @ torch.randn(32, 32, device=device)
    loss.backward()
    torch.cuda.synchronize()
    require(bool(torch.isfinite(y).all()) and bool(torch.isfinite(conv.weight.grad).all()), 'CUDA kernel check failed')
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(device), capability=[major, minor],
                         cuda=torch.version.cuda, architectures=torch.cuda.get_arch_list(),
                         versions=runtime_versions(), cuda_bf16_kernel_check='PASS', trained=False), indent=2))


if __name__ == '__main__':
    main()
