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


def test_modules():
    pre = Preprocess(mean=0.2860, std=0.3530)
    x = torch.tensor([0, 255], dtype=torch.uint8).reshape(1,1,1,2)
    out = pre(x)
    print(out.dtype, out.numpy())
    assert out.dtype == torch.float32
    assert torch.allclose(out.flatten(), torch.tensor([-0.2860 / 0.3530, (1 - 0.2860) / 0.3530]))

    logits = torch.randn(2, 10, dtype=torch.float32)
    post = Postprocess()
    out = post(logits)
    print(out[0].shape, out[1].shape)
    assert torch.allclose(out[0], torch.argmax(out[1], dim=1))


if __name__ == '__main__':
    test_modules()
