#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from openvino.runtime import Core
import argparse
import os
import time  # ★追加：時間計測用

def run_inference(model_xml, input_img, output_dir="ov_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ==== 全体時間スタート ====
    t_total_start = time.perf_counter()

    # OpenVINO Runtime
    core = Core()
    model = core.read_model(model_xml)
    compiled_model = core.compile_model(model, device_name="CPU")

    # 入力ノード
    input_layer = compiled_model.input(0)
    n, c, h, w = input_layer.shape  # たぶん (1,3,512,512)

    # 画像読み込み（元サイズも保持しておく）
    orig_bgr = cv2.imread(input_img)
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_img}")
    orig_h, orig_w = orig_bgr.shape[:2]

    # 推論用にリサイズ
    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (w, h))

    # Normalize (ImageNet)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_input = img_norm.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)

    # ==== 推論だけの時間計測 ====
    t_inf_start = time.perf_counter()
    result = compiled_model([img_input])[compiled_model.output(0)]
    t_inf_end = time.perf_counter()

    result = result[0, 0]  # (H,W)

    # Sigmoid → 0/255 mask (512x512)
    mask = 1 / (1 + np.exp(-result))
    mask_bin = (mask > 0.5).astype(np.uint8) * 255  # (h,w)

    # マスクを元画像サイズにリサイズ
    mask_bin_resized = cv2.resize(
        mask_bin, (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    # overlay は元画像サイズで作る
    overlay = orig_bgr.copy()
    overlay[mask_bin_resized == 255] = (0, 255, 0)  # 緑で道路

    # 保存
    base = os.path.splitext(os.path.basename(input_img))[0]
    mask_path = f"{output_dir}/{base}_mask.png"
    overlay_path = f"{output_dir}/{base}_overlay.png"
    cv2.imwrite(mask_path, mask_bin_resized)
    cv2.imwrite(overlay_path, overlay)

    # ==== 全体時間エンド ====
    t_total_end = time.perf_counter()

    # ==== 時間の表示 ====
    inf_ms = (t_inf_end - t_inf_start) * 1000.0
    total_ms = (t_total_end - t_total_start) * 1000.0

    print("Saved:")
    print(f" - {mask_path}")
    print(f" - {overlay_path}")
    print(f"\n=== Time ===")
    print(f"  Inference only : {inf_ms:.2f} ms")
    print(f"  Total (load + pre + post) : {total_ms:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_xml", required=True, help="OpenVINO XML model")
    parser.add_argument("--input_img", required=True, help="input image")
    parser.add_argument("--output_dir", default="ov_outputs")
    args = parser.parse_args()

    run_inference(args.model_xml, args.input_img, args.output_dir)
