"""Unpaired image dataset (CycleGAN format: trainA/, trainB/, testA/, testB/).$"""
from pathlib import Path
import numpy as np
from PIL import Image


_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_image(path: str | Path, size: int) -> np.ndarray:
    """Load an image as (3, H, W) float32 in [-1, 1]"""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """(3, H, W) in [-1, 1] -> (H, W, 3) uint8 for saving"""
    img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return img.transpose(1, 2, 0)


class UnpairedDataset:
    """Two folders sampled independently. One epoch = max(len A, len B) iterations"""

    def __init__(self, dirA: str, dirB: str, size: int = 64, max_per_side: int | None = None):
        self.size = size
        self.A = sorted(str(p) for p in Path(dirA).iterdir() if p.suffix.lower() in _EXTS)
        self.B = sorted(str(p) for p in Path(dirB).iterdir() if p.suffix.lower() in _EXTS)
        if not self.A or not self.B:
            raise RuntimeError(f"Empty dataset: {dirA} ({len(self.A)}) / {dirB} ({len(self.B)})")
        if max_per_side:
            self.A = self.A[:max_per_side]
            self.B = self.B[:max_per_side]

    def __len__(self) -> int:
        return max(len(self.A), len(self.B))

    def epoch(self):
        """Yield (a, b) pairs, each of shape (1, 3, size, size), independently shuffled"""
        a_idx = np.random.permutation(len(self.A))
        b_idx = np.random.permutation(len(self.B))
        n = len(self)
        for i in range(n):
            a = load_image(self.A[a_idx[i % len(self.A)]], self.size)
            b = load_image(self.B[b_idx[i % len(self.B)]], self.size)
            yield a[None], b[None]
