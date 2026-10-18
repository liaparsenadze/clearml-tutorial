import argparse
import onnx
import torch
from torch.export import Dim
from pathlib import Path
from model import CNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    parser.add_argument("-m", "--model-name", type=str, required=True)
    args = parser.parse_args()

    return args.input_path, args.output_path, args.model_name


def convert(input_path, output_path, model_name):
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


def main(input_path, output_path, model_name):
    onnx_path = convert(input_path, output_path, model_name)
    verify(onnx_path)


if __name__ == '__main__':
    main(*parse_args())