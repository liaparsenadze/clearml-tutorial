import tritonclient.http as httpclient
from PIL import Image
from train import normalize
import torch
import numpy as np


def infer_img(client, img_path):
    img = Image.open(img_path)
    img = normalize(torch.from_numpy(np.array(img)[None]), 0.286, 0.353).numpy()
    input_tensor = httpclient.InferInput(
        "input",
        img.shape,
        "FP32",
    )
    input_tensor.set_data_from_numpy(img)
    resp = client.infer(
        "fashion-v1",
        model_version='1',
        inputs=[input_tensor]
    )

    return  np.argmax(resp.as_numpy("logits"))


if __name__ == '__main__':
    client = httpclient.InferenceServerClient(url="localhost:8000")
    print(infer_img(client, '/Users/lparsenadze/github/clearml-tutorial/fashion_mnist_data/0.png'))
    client.close()
