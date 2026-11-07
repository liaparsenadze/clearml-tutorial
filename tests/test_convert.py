import pytest
import numpy as np
import torch
import convert as c
import onnxruntime as ort
from deploy_processors import Preprocess, Postprocess
from model import CNN


def test_convert_model_creates_onnx_file(model_path, tmp_path):
    onnx_path = c.convert_model(model_path, output_path=tmp_path, model_name="test_model")
    assert onnx_path == tmp_path / "test_model" / "1" / "model.onnx"
    assert onnx_path.is_file()



@pytest.mark.parametrize(
    "fn, module_name, module",
    [
        (c.convert_preprocess_module, "pre", Preprocess(mean=0.2860, std=0.3530)),
        (c.convert_postprocess_module, "post", Postprocess())
    ]
)
def test_convert_module_creates_onnx_file(fn, module_name, module, tmp_path):
    onnx_path = fn(module, tmp_path, module_name)
    assert onnx_path == tmp_path / module_name / "1" / "model.onnx"
    assert onnx_path.is_file()


def test_convert_preprocess_creates_correct_onnx_file(tmp_path):
    module = Preprocess(mean=0.2860, std=0.3530)
    img = (np.arange(4 * 28 * 28) % 256).astype(np.uint8).reshape(4, 1, 28, 28)

    onnx_path = c.convert_preprocess_module(module, tmp_path, "pre")
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"input": img})[0]

    with torch.no_grad():
        torch_out = module(torch.from_numpy(img)).numpy()

    np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-6, atol=1e-6)


def test_convert_model_creates_correct_onnx_file(model_path, tmp_path):
    model = CNN(in_channels=1, n_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img = torch.randn(4, 1, 28, 28)

    onnx_path = c.convert_model(model_path, output_path=tmp_path, model_name="test_model")
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"input": img.numpy()})[0]

    with torch.no_grad():
        torch_out = model(img).numpy()

    np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-5, atol=1e-5)


def test_convert_postprocess_creates_correct_onnx_file(tmp_path):
    module = Postprocess()
    pred = torch.randn(4, 10)

    onnx_path = c.convert_postprocess_module(module, tmp_path, "post")
    sess = ort.InferenceSession(onnx_path)
    onnx_cls, onnx_conf = sess.run(None, {"logits": pred.numpy()})
    with torch.no_grad():
        torch_cls, torch_conf = module(pred)

    np.testing.assert_allclose(onnx_cls, torch_cls.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(onnx_cls, torch_cls.numpy(), rtol=1e-6, atol=1e-6)
