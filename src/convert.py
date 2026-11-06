import argparse
import onnx
import torch
from torch.export import Dim
from pathlib import Path
from model import CNN
from pydantic import BaseModel
from enum import StrEnum
from google.protobuf import text_format, json_format
from tritonclient.grpc.model_config_pb2 import ModelConfig
from deploy_processors import Preprocess, Postprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    parser.add_argument("-m", "--model-name", type=str, required=True)
    args = parser.parse_args()

    return args.input_path, args.output_path, args.model_name


class DataType(StrEnum):
    FP32 = "TYPE_FP32"
    UINT8 = "TYPE_UINT8"


class _Dim(BaseModel):
    name: str
    dims: list[int]
    data_type: DataType = DataType.FP32


class FashionModelConfig(BaseModel):
    name: str
    backend: str = "onnxruntime"
    max_batch_size: int = 32
    input: list[_Dim]
    output: list[_Dim]


def convert_preprocess_module(module, output_path, module_name):
    module.eval()
    example_input = torch.randint(0, 256, (1, 1, 28, 28), dtype=torch.uint8)
    onnx_program = torch.onnx.export(
        module,
        example_input,
        dynamic_shapes={"x": {0: Dim("batch")}},
        input_names=['input'],
        output_names=['norm_input'],
    )
    model_dir = output_path / module_name / "1"
    onnx_path = model_dir / "model.onnx"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_program.save(onnx_path)

    return onnx_path


def convert_model(input_path, output_path, model_name):
    model = CNN(in_channels=1, n_classes=10)
    weights = torch.load(input_path)
    model.load_state_dict(weights)
    model.eval()
    example_input = torch.randn(1, 1, 28, 28)
    onnx_program = torch.onnx.export(
        model,
        example_input,
        dynamic_shapes={"x": {0: Dim("batch")}},
        input_names=["input"],
        output_names=["logits"]
    )
    model_dir = output_path / model_name / "1"
    onnx_path = model_dir / "model.onnx"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_program.save(onnx_path)

    return onnx_path


def verify(onnx_path):
    onnx_program = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_program)


def generate_preprocess_config(module_name, output_path):
    config = FashionModelConfig(
        name=module_name,
        input=[_Dim(name="input", dims=[1,28,28], data_type=DataType.UINT8)],
        output=[_Dim(name="norm_input", dims=[1,28,28])],
    )

    pb = json_format.ParseDict(config.model_dump(), ModelConfig())
    pbtxt = text_format.MessageToString(pb, use_short_repeated_primitives=True)

    with open(output_path / module_name / "config.pbtxt", "w") as f:
        f.write(pbtxt)


def generate_model_config(model_name, output_path):
    config = FashionModelConfig(
        name=model_name,
        input=[_Dim(name='input', dims=[1,28,28])],
        output=[_Dim(name='logits', dims=[10])]
    )
    pb = json_format.ParseDict(config.model_dump(), ModelConfig())
    pbtxt = text_format.MessageToString(pb, use_short_repeated_primitives=True)

    with open(output_path / f"{model_name}/config.pbtxt", "w") as f:
        f.write(pbtxt)


def main(input_path, output_path, model_name):
    module_name = f"preprocess_{model_name}"
    onnx_path = convert_preprocess_module(Preprocess(mean=0.2860, std=0.3530), output_path, module_name)
    verify(onnx_path)
    generate_preprocess_config(module_name, output_path)

    onnx_path = convert_model(input_path, output_path, model_name)
    verify(onnx_path)
    generate_model_config(model_name, output_path)




if __name__ == '__main__':
    main(*parse_args())
