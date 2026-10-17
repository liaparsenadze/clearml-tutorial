from pathlib import Path

import numpy as np


def load_idx_images(path: Path) -> np.ndarray:
    # IDX image file: 16-byte header, then one uint8 per pixel
    return np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def load_idx_labels(path: Path) -> np.ndarray:
    # IDX label file: 8-byte header, then one uint8 label per image
    return np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=8)

