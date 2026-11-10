import tempfile
from pathlib import Path

from clearml import Model
from utils import parse_arguments, setup_task
import convert as c
from triton_ensemble import generate_ensemble_config
from convert import ConvertSpec
import json


def convert_to_onnx(weights_path, spec: ConvertSpec, output_path):
    # preprocess conversion
    preprocess_onnx_path = c.convert_preprocess_module(spec, output_path)
    c.verify(preprocess_onnx_path)
    c.generate_preprocess_config(spec, output_path)

    # model conversion
    model_onnx_path = c.convert_model(weights_path, spec, output_path)
    c.verify(onnx_path=model_onnx_path)
    c.generate_model_config(spec, output_path)

    # postprocess conversion
    postprocess_onnx_path = c.convert_postprocess_module(spec, output_path)
    c.verify(postprocess_onnx_path)
    c.generate_postprocess_config(spec, output_path)


def deploy(project_name, task_name, config_path, queue_name):
    task, config = setup_task(project_name, task_name, config_path, queue_name)

    model = Model(model_id=config['model_id'])
    metadata = json.loads(model.get_metadata("config"))
    weights_path = model.get_local_copy()

    spec = ConvertSpec(
        name=config["name"],
        mean=float(metadata["mean"]),
        std=float(metadata["std"]),
        input_shape=config["input_shape"],
        n_classes=len(model.labels)
    )

    with tempfile.TemporaryDirectory() as output_path:
        convert_to_onnx(weights_path, spec, Path(output_path))
        generate_ensemble_config(spec, Path(output_path))
        task.upload_artifact(name='model_repository', artifact_object=output_path, wait_on_upload=True)

    task.close()


if __name__ == "__main__":
    deploy(*parse_arguments())