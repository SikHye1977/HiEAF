"""Wrapper for an unmodified, pinned official AASIST checkout."""
from __future__ import annotations

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from threading import RLock

import torch
from torch import Tensor, nn

from .interfaces import ModelOutput

_IMPORT_LOCK = RLock()


class AASISTWrapper(nn.Module):
    """Expose official embedding/logits and a configured fake probability."""

    def __init__(self, source_root: Path, model_args: dict, *, fake_class_index: int):
        super().__init__()
        if fake_class_index not in (0, 1):
            raise ValueError("Official AASIST has two class logits; fake_class_index must be 0 or 1")
        self.fake_class_index = fake_class_index
        self.backbone = _load_official_model(source_root, model_args)

    def forward(self, waveform: Tensor) -> ModelOutput:
        embedding, logits = self.backbone(waveform)
        fake_probability = torch.softmax(logits, dim=-1)[..., self.fake_class_index]
        return ModelOutput(features=embedding, logits=logits, fake_probability=fake_probability)


def _load_official_model(source_root: Path, model_args: dict) -> nn.Module:
    """Load the verified vendor file without claiming the generic `models` name."""
    source_root = Path(source_root).resolve()
    source = source_root / "models" / "AASIST.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official AASIST source missing: {source_root}")
    # AASIST.py is self-contained. Never import the script's sibling models.py,
    # even when it is first on sys.path or already present in sys.modules.
    identity = hashlib.sha256(str(source).encode("utf-8") + source.read_bytes()).hexdigest()
    name = "_artifact_emotion_aasist_" + identity
    with _IMPORT_LOCK:
        module = sys.modules.get(name)
        if module is None:
            spec = spec_from_file_location(name, source)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load official AASIST source: {source}")
            module = module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
    return module.Model(model_args)


def build_aasist(source_root: Path, model_args: dict, *, fake_class_index: int) -> AASISTWrapper:
    """Build without loading an official checkpoint or any waveform data."""
    return AASISTWrapper(source_root, model_args, fake_class_index=fake_class_index)
