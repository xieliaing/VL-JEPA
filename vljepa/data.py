"""Minimal, self-contained dataset for VL-JEPA image/video-text manifests.

JSONL rows (one per line):
    {"image": "path.jpg", "query": "Describe this image.", "target": "a caption"}
    {"video": "clip.mp4", "query": "...", "target": "..."}   # video needs PyAV

Frames are returned as [T, C, H, W]; images are duplicated across T frames.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

try:
    import av as _av  # noqa: F401 — optional; only for video rows
    _AV = True
except ImportError:
    _AV = False


class VisionLanguageJsonlDataset(Dataset):
    def __init__(self, manifest_path: str | Path, num_frames: int = 1, image_size: int = 224) -> None:
        if num_frames <= 0 or image_size <= 0:
            raise ValueError("num_frames and image_size must be > 0")
        self.path = Path(manifest_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.path}")
        self.num_frames = num_frames
        # ImageNet normalization (standard for ViT-family vision encoders).
        self.transform = v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        with open(self.path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        if not self.samples:
            raise ValueError(f"Manifest has no rows: {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _image_frames(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        frame = self.transform(img)                         # [C, H, W]
        return frame.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)  # [T, C, H, W]

    def _video_frames(self, video_path: str) -> torch.Tensor:
        if not _AV:
            raise ImportError("PyAV required for video rows: pip install av")
        import av
        with av.open(str(video_path)) as container:
            raw = [f.to_image() for f in container.decode(video=0)]
        if not raw:
            raise RuntimeError(f"Empty video: {video_path}")
        idx = torch.linspace(0, len(raw) - 1, steps=self.num_frames).long().tolist()
        return torch.stack([self.transform(raw[i].convert("RGB")) for i in idx], dim=0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.samples[index]
        if "target" not in row:
            raise KeyError(f"Row {index} missing required `target` field.")
        if row.get("video"):
            frames = self._video_frames(row["video"])
        elif row.get("image"):
            frames = self._image_frames(row["image"])
        else:
            raise ValueError("Each row must have `image` or `video`.")
        return {"frames": frames, "query": row.get("query", ""), "target": row["target"]}
