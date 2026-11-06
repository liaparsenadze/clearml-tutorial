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
        return torch.argmax(x, dim=1), torch.softmax(x, dim=1)
