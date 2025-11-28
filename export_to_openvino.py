#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_to_openvino.py
- deeplabv3plus_road_best.pth -> deeplabv3plus_road.onnx (legacy exporter)
- deeplabv3plus_road.onnx -> OpenVINO IR (xml+bin)
※ onnx_env など、openvino-dev と smp が入っている環境で実行
"""

import os
import subprocess

import torch
import segmentation_models_pytorch as smp


CKPT_PATH = "checkpoints/deeplabv3plus_road_best.pth"
ONNX_PATH = "deeplabv3plus_road.onnx"
OPENVINO_DIR = "openvino_model"
OPENVINO_MODEL = os.path.join(OPENVINO_DIR, "deeplabv3plus_road.xml")


def get_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx():
    print("=== Exporting to ONNX (legacy exporter) ===")
    model = get_model()
    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"Saved ONNX: {ONNX_PATH}")


def convert_to_openvino():
    print("=== ONNX → OpenVINO IR ===")
    os.makedirs(OPENVINO_DIR, exist_ok=True)

    cmd = [
        "ovc",
        ONNX_PATH,
        "--output_model",
        OPENVINO_MODEL,
    ]
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[SUCCESS] OpenVINO model saved to:", OPENVINO_MODEL)


def main():
    export_onnx()
    convert_to_openvino()


if __name__ == "__main__":
    main()
