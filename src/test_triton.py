import tritonclient.http as httpclient
from PIL import Image
from  transforms import normalize
import torch
import numpy as np


def infer_img(client, img_path):
    img = np.array(Image.open(img_path))
    img = img[None, None]
    input_tensor = httpclient.InferInput(
        "input",
        img.shape,
        "UINT8",
    )
    input_tensor.set_data_from_numpy(img)
    resp = client.infer(
        "ensemble_fashion-v1",
        model_version='1',
        inputs=[input_tensor]
    )

    return  resp.as_numpy('class_id'), resp.as_numpy('conf')


if __name__ == '__main__':
    client = httpclient.InferenceServerClient(url="localhost:8000")
    print(infer_img(client, '/Users/lparsenadze/github/clearml-tutorial/0.png'))
    client.close()
