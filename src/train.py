import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from clearml import Dataset
from pathlib import Path
from utils import setup_task, parse_arguments
import numpy as np
from model import CNN
import json


def normalize(img_tensor, label_tensor, mean, std):
    img_tensor = img_tensor.float() / 255.0
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(1), label_tensor.long()


def train(project_name, task_name, config_path, queue_name):
    task, config = setup_task(project_name, task_name, config_path, queue_name)
    dataset_id = config["dataset_id"]

    data_dir = Path(Dataset.get(dataset_id=dataset_id).get_local_copy())
    train_data = np.load(data_dir / 'train.npz')

    dataset = TensorDataset(torch.tensor(train_data['images']), torch.tensor(train_data['labels']))

    model = CNN(in_channels=1, n_classes=10)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(config['epochs']):
        _loss = 0
        for i, batch in enumerate(loader):
            normed_batch = normalize(*batch, config['mean'], config['std'])
            opt.zero_grad()
            pred = model(normed_batch[0])
            loss = F.cross_entropy(pred, normed_batch[1])
            loss.backward()
            opt.step()

            task.get_logger().report_scalar("loss", 'train', loss.item(), iteration=epoch * len(loader) + i)
            _loss += loss.item()
        print(f"Loss: {_loss / len(loader):.4f}")

    task.set_model_label_enumeration({
        "T-shirt/top": 0, "Trouser": 1, "Pullover": 2, "Dress": 3, "Coat": 4,
        "Sandal": 5, "Shirt": 6, "Sneaker": 7, "Bag": 8, "Ankle boot": 9,
    })
    # save
    torch.save(model.state_dict(), "model.pth")
    output_model = task.models["output"][-1]
    output_model.set_metadata("config", json.dumps(config))


if __name__ == "__main__":
    train(*parse_arguments())