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


class DataType(StrEnum):
    FP32 = "TYPE_FP32"
    UINT8 = "TYPE_UINT8"
    INT64 = "TYPE_INT64"


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


def convert_preprocess_module(clearml_model, metadata, output_path):
    module = Preprocess(mean=metadata.mean, std=metadata.std)
    module.eval()
    example_input = torch.randint(0, 256, (1, *metadata.input_shape), dtype=torch.uint8)
    onnx_program = torch.onnx.export(
        module,
        example_input,
        dynamic_shapes={"x": {0: Dim("batch")}},
        input_names=['input'],
        output_names=['norm_input'],
    )
    model_dir = output_path / f"preprocess_{clearml_model.name}" / "1"
    onnx_path = model_dir / "model.onnx"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_program.save(onnx_path)

    return onnx_path


def convert_model(output_path, clearml_model, metadata):
    model = CNN(in_channels=metadata.input_shape[0], n_classes=len(clearml_model.labels))
    weights = torch.load(clearml_model.get_local_copy())
    model.load_state_dict(weights)
    model.eval()
    example_input = torch.randn(1, *metadata.input_shape)
    onnx_program = torch.onnx.export(
        model,
        example_input,
        dynamic_shapes={"x": {0: Dim("batch")}},
        input_names=["input"],
        output_names=["logits"]
    )
    model_dir = output_path / clearml_model.name / "1"
    onnx_path = model_dir / "model.onnx"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_program.save(onnx_path)

    return onnx_path


def convert_postprocess_module(clearml_model, output_path):
    module = Postprocess()
    module.eval()
    example_input = torch.rand(1, len(clearml_model.labels))
    onnx_program = torch.onnx.export(
        module,
        example_input,
        dynamic_shapes={"x": {0: Dim("batch")}},
        input_names=['logits'],
        output_names=['class_id', 'conf']
    )
    model_dir = output_path / f"postprocess_{clearml_model.name}" / "1"
    onnx_path = model_dir / "model.onnx"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_program.save(onnx_path)

    return onnx_path


def verify(onnx_path):
    onnx_program = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_program)


def generate_preprocess_config(clearml_model, metadata, output_path):
    config = FashionModelConfig(
        name=f"preprocess_{clearml_model.name}",
        input=[_Dim(name="input", dims=metadata.input_shape, data_type=DataType.UINT8)],
        output=[_Dim(name="norm_input", dims=metadata.input_shape)],
    )

    pb = json_format.ParseDict(config.model_dump(), ModelConfig())
    pbtxt = text_format.MessageToString(pb, use_short_repeated_primitives=True)

    with open(output_path / clearml_model.name / "config.pbtxt", "w") as f:
        f.write(pbtxt)


def generate_model_config(clearml_model, metadata, output_path):
    config = FashionModelConfig(
        name=clearml_model.name,
        input=[_Dim(name='input', dims=metadata.input_shape)],
        output=[_Dim(name='logits', dims=[len(clearml_model.labels)])]
    )
    pb = json_format.ParseDict(config.model_dump(), ModelConfig())
    pbtxt = text_format.MessageToString(pb, use_short_repeated_primitives=True)

    with open(output_path / f"{clearml_model.name}/config.pbtxt", "w") as f:
        f.write(pbtxt)


def generate_postprocess_config(clearml_model, output_path):
    config = FashionModelConfig(
        name=clearml_model.name,
        input=[_Dim(name='logits', dims=[len(clearml_model.labels)])],
        output=[_Dim(name='class_id', dims=[1], data_type=DataType.INT64), _Dim(name='conf', dims=[1])],
    )
    pb = json_format.ParseDict(config.model_dump(), ModelConfig())
    pbtxt = text_format.MessageToString(pb, use_short_repeated_primitives=True)

    with open(output_path / f"postprocess_{clearml_model.name}/config.pbtxt", "w") as f:
        f.write(pbtxt)
