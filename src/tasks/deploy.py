import tempfile
from pathlib import Path

from clearml import Model

from utils import parse_arguments, setup_task
import convert as c
from triton_ensemble import generate_ensemble_config


def convert_to_onnx(clearml_model, metadata, output_path):
    '''metadata=model metadata'''
    # preprocess conversion
    preprocess_onnx_path = c.convert_preprocess_module(clearml_model, metadata, output_path)
    c.verify(preprocess_onnx_path)
    c.generate_preprocess_config(clearml_model, metadata, output_path)

    # model conversion
    model_onnx_path = c.convert_model(output_path, clearml_model, metadata)
    c.verify(onnx_path=model_onnx_path)
    c.generate_model_config(clearml_model, metadata, output_path)

    # postprocess conversion
    postprocess_onnx_path = c.convert_postprocess_module(clearml_model, output_path)
    c.verify(postprocess_onnx_path)
    c.generate_postprocess_config(clearml_model, output_path)


def deploy(project_name, task_name, config_path, queue_name):
    task, config = setup_task(project_name, task_name, config_path, queue_name)
    model_id = config["model_id"]
    model = Model(model_id=model_id)
    metadata = model.get_metadata("config")

    # what is output_path
    # should it be some sort of temp file on worker side?
    with tempfile.TemporaryDirectory() as output_path:
        convert_to_onnx(model, metadata, Path(output_path))
        generate_ensemble_config(model, metadata, Path(output_path))
        task.upload_artifact(name=model.name, artifact_object=output_path, wait_on_upload=True)

    task.close()


if __name__ == "__main__":
    deploy(*parse_arguments())