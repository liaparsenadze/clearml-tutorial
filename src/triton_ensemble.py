from pydantic import BaseModel
import argparse

from convert import _Dim, DataType
from pathlib import Path
from google.protobuf import text_format, json_format
from tritonclient.grpc.model_config_pb2 import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    parser.add_argument("-m", "--model-name", type=str, required=True)
    args = parser.parse_args()

    return args.model_name, args.output_path


class EnsembleStepConfig(BaseModel):
    model_name: str
    model_version: int
    input_map: dict[str, str]
    output_map: dict[str, str]

class EnsembleConfig(BaseModel):
    name: str
    platform: str
    max_batch_size: int
    input: list[_Dim]
    output: list[_Dim]
    ensemble_scheduling: dict[str, list[EnsembleStepConfig]]


def generate_ensemble_config(model_name:str, output_path: Path):
    config = EnsembleConfig(
        name=f"ensemble_{model_name}",
        platform="ensemble",
        max_batch_size=32,
        input=[_Dim(name='input', dims=[1,28,28], data_type=DataType.UINT8)],
        output=[
            _Dim(name='class_id', dims=[1], data_type=DataType.INT64),
            _Dim(name='conf', dims=[1])
        ],
        ensemble_scheduling={
            "step": [
                EnsembleStepConfig(
                    model_name=f"preprocess_{model_name}",
                    model_version=1,
                    input_map={"input": "input"},
                    output_map={"norm_input": "norm_input"},
                ),
                EnsembleStepConfig(
                    model_name=f"{model_name}",
                    model_version=1,
                    input_map={"input": "norm_input"},
                    output_map={"logits": "logits"},
                ),
                EnsembleStepConfig(
                    model_name=f"postprocess_{model_name}",
                    model_version=1,
                    input_map={"logits": "logits"},
                    output_map={"class_id": "class_id", "conf": "conf"},
                )
            ]
        }
    )
    pb = json_format.ParseDict(config.model_dump(), ModelConfig())
    pbtxt = text_format.MessageToString(pb, use_short_repeated_primitives=True)
    (output_path / f"ensemble_{model_name}/1").mkdir(parents=True, exist_ok=True)

    with open(output_path / f"ensemble_{model_name}/config.pbtxt", "w") as f:
        f.write(pbtxt)


def main(model_name: str, output_path: Path):
    generate_ensemble_config(model_name, output_path)


if __name__ == "__main__":
    main(*parse_args())
