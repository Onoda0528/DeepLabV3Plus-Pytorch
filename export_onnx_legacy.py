#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx_legacy.py
- segmentation_models_pytorch の DeepLabV3Plus(resnet34) を
  "古い" ONNX exporter で deeplabv3plus_road.onnx に出力するだけ
- onnx_env などの専用環境で実行
"""

import torch
import segmentation_models_pytorch as smp


CKPT_PATH = "checkpoints/deeplabv3plus_road_best.pth"
ONNX_PATH = "deeplabv3plus_road.onnx"


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


def main():
    print("=== Exporting to ONNX (legacy exporter) ===")
    model = get_model()

    dummy_input = torch.randn(1, 3, 512, 512)  # 学習時と同じ解像度

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


if __name__ == "__main__":
    main()
