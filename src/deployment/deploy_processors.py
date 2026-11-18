import torch
from torch import nn


class Preprocess(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return (x - self.mean) / self.std


class Postprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(x, dim=1)
        score, class_id = torch.max(probs, dim=1, keepdim=True)
        return class_id, score
