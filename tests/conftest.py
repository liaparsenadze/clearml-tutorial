import pytest
import torch
import torchvision.io as io
from model import CNN


@pytest.fixture
def model_path(tmp_path):
    model = CNN(in_channels=1, n_classes=10)
    torch.save(model.state_dict(), str(tmp_path / "model.pt"))
    return str(tmp_path / "model.pt")

@pytest.fixture
def image_path(tmp_path):
    img = torch.randint(low=0, high=256, size=(1, 28, 28), dtype=torch.uint8)
    io.write_png(img, str(tmp_path / "image.png"))
    return str(tmp_path / "image.png")
