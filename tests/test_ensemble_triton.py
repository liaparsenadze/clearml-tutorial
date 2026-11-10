import tritonclient.http as httpclient
from PIL import Image
from convert import ConvertSpec
import pytest

from deploy_processors import Postprocess, Preprocess
from model import CNN
import torch
import numpy as np


def infer_img_from_triton(spec, client, img_path):
    img = np.array(Image.open(img_path))
    img = img[None, None]
    input_tensor = httpclient.InferInput(
        "input",
        img.shape,
        "UINT8",
    )
    input_tensor.set_data_from_numpy(img)
    resp = client.infer(
        f"ensemble_{spec.name}",
        model_version='1',
        inputs=[input_tensor]
    )

    return  resp.as_numpy('class_id'), resp.as_numpy('conf')

def infer_img_from_torch(spec, model_path, img_path):
    model = CNN(in_channels=spec.input_shape[0], n_classes=spec.n_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    pre = Preprocess(mean=spec.mean, std=spec.std)
    post = Postprocess()
    img = np.array(Image.open(img_path))
    img = torch.from_numpy(img[None, None])

    with torch.no_grad():
        x = pre(img)
        x = model(x)
        class_id, conf = post(x)

    return class_id.numpy(), conf.numpy()



def test_ensemble_triton():
    client = httpclient.InferenceServerClient(url="localhost:8000")
    try:
        ready = client.is_server_ready()
    except Exception as e:
        ready = False
    if not ready:
        pytest.skip("Triton not reachable")

    spec = ConvertSpec(name='fashion-v1', mean=0.2860, std=0.3530, input_shape=[1, 28, 28], n_classes=10)
    triton_res = infer_img_from_triton(spec, client, '/Users/lparsenadze/github/clearml-tutorial/0.png')
    client.close()
    torch_res = infer_img_from_torch(spec, '/Users/lparsenadze/github/clearml-tutorial/fashion_mnist_data/model.pth', '/Users/lparsenadze/github/clearml-tutorial/0.png')

    np.testing.assert_allclose(triton_res[0], torch_res[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(triton_res[1], torch_res[1], rtol=1e-5, atol=1e-5)
