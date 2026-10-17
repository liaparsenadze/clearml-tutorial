import tempfile
import numpy as np

from transforms import load_idx_images, load_idx_labels
from utils import setup_task, parse_arguments
from pathlib import Path
from clearml import Dataset, Task


def main(project_name, task_name, config_path, queue_name):
    task, config = setup_task(project_name, task_name, config_path, queue_name, Task.TaskTypes.data_processing)
    raw_id = config['dataset_id']
    raw = Path(Dataset.get(dataset_id=raw_id).get_local_copy())

    out = Path(tempfile.mkdtemp())
    for split, prefix in [("train", "train"), ("test", "t10k")]:
        np.savez_compressed(out / f"{split}.npz",
                            images=load_idx_images(raw / f"{prefix}-images-idx3-ubyte"),
                            labels=load_idx_labels(raw / f"{prefix}-labels-idx1-ubyte"),
                            )
    dataset = Dataset.create(
        dataset_name = config["dataset_name"],
        dataset_project=project_name,
        parent_datasets=[raw_id]
    )
    dataset.add_files(out)
    dataset.upload()
    dataset.finalize()

if __name__ == '__main__':
    main(*parse_arguments())