#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_road.py
- 学習済み DeepLabV3Plus(resnet34) で道路マスク推論
- 入力: 画像の glob パターン
- 出力: mask と overlay を out_dir に保存
"""

import os
import glob
import argparse

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


CKPT_PATH_DEFAULT = "checkpoints/deeplabv3plus_road_best.pth"


def get_model(ckpt_path, device):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,  # checkpoint から読むので None でOK
        in_channels=3,
        classes=1,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_bgr):
    """BGR画像をモデル入力 (1,3,512,512) tensor に変換"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (512, 512))

    img = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
    tensor = torch.from_numpy(img).unsqueeze(0)  # (1,3,512,512)
    return tensor, (h, w)


def postprocess_mask(logits, orig_size, threshold=0.5):
    """ロジット -> 0/255 マスク (元の解像度)"""
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H,W)
    mask = (probs > threshold).astype(np.uint8) * 255  # 0 or 255
    mask = cv2.resize(mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def make_overlay(img_bgr, mask, alpha=0.5):
    """もとの画像に赤マスクを重ねる"""
    color = np.zeros_like(img_bgr, dtype=np.uint8)
    color[:, :, 2] = 255  # 赤

    mask_bool = mask.astype(bool)
    overlay = img_bgr.copy()
    overlay[mask_bool] = cv2.addWeighted(
        img_bgr[mask_bool], 1 - alpha, color[mask_bool], alpha, 0
    )
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        required=True,
        help='入力画像のglobパターン (例: "test_images/*.png")',
    )
    parser.add_argument(
        "--ckpt",
        default=CKPT_PATH_DEFAULT,
        help="学習済みモデルのパス",
    )
    parser.add_argument(
        "--out_dir",
        default="road_outputs",
        help="出力ディレクトリ (mask と overlay を保存)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("loading model from:", args.ckpt)
    model = get_model(args.ckpt, device)

    img_paths = sorted(glob.glob(args.input_glob))
    if not img_paths:
        print(f"[!] no images matched: {args.input_glob}")
        return

    print(f"{len(img_paths)} images found.")

    for img_path in img_paths:
        print("processing:", img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[!] failed to read: {img_path}")
            continue

        inp, orig_size = preprocess_image(img_bgr)
        inp = inp.to(device)

        with torch.no_grad():
            logits = model(inp)  # (1,1,512,512)

        mask = postprocess_mask(logits, orig_size)
        overlay = make_overlay(img_bgr, mask)

        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(args.out_dir, f"{base}_mask.png")
        overlay_path = os.path.join(args.out_dir, f"{base}_overlay.png")

        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)

        print("  saved:", mask_path)
        print("  saved:", overlay_path)


if __name__ == "__main__":
    main()
