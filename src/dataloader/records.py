from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    sample_id: str
    media_path: Path
    split: str
    clip_label: int | None
    video_label: int | None
    audio_label: int | None
    group_id: str | None = None
