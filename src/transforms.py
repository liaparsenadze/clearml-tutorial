from pathlib import Path

import numpy as np


def load_idx_images(path: Path) -> np.ndarray:
    # IDX image file: 16-byte header, then one uint8 per pixel
    return np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def load_idx_labels(path: Path) -> np.ndarray:
    # IDX label file: 8-byte header, then one uint8 label per image
    return np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=8)

def normalize(img_tensor, mean, std, label_tensor=None):
    img_tensor = img_tensor.float() / 255.0
    img_tensor = (img_tensor - mean) / std
    if label_tensor:
        return img_tensor.unsqueeze(1), label_tensor.long()
    return img_tensor.unsqueeze(1)
