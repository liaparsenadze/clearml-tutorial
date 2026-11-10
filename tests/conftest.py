import pytest
import torch
import torchvision.io as io
from model import CNN
from convert import ConvertSpec


@pytest.fixture
def convert_spec():
    return ConvertSpec(
        name="test_model",
        mean=0.2,
        std=0.3,
        input_shape=[1,28,28],
        n_classes=10
    )


@pytest.fixture
def model_path(convert_spec, tmp_path):
    model = CNN(in_channels=convert_spec.input_shape[0], n_classes=convert_spec.n_classes)
    torch.save(model.state_dict(), str(tmp_path / "model.pt"))
    return str(tmp_path / "model.pt")

@pytest.fixture
def image_path(convert_spec, tmp_path):
    img = torch.randint(low=0, high=256, size=convert_spec.input_shape, dtype=torch.uint8)
    io.write_png(img, str(tmp_path / "image.png"))
    return str(tmp_path / "image.png")
