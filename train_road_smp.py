#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_road_smp.py
- segmentation_models_pytorch の DeepLabV3Plus(resnet34) を使った
  2値（道路 vs 背景）セグメンテーション学習スクリプト
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# =========================
# データセット
# =========================

class RoadDataset(Dataset):
    """
    data/images/*.png と data/masks/*.png を読む2値セグメンテーション用Dataset
    - image: RGB
    - mask : 0 or 1（道路=1, 背景=0）
    """
    def __init__(self, images_dir, masks_dir, transforms=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        assert len(self.image_paths) == len(self.mask_paths), "画像とマスクの枚数が違います"
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 画像: BGR -> RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # マスク: グレースケール
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 0 or 1 に正規化（255白マスク前提）
        mask = (mask > 127).astype("float32")  # (H, W)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]          # (3, H, W) float32 (torch.Tensor)
            mask = augmented["mask"]           # (H, W) float32 (torch.Tensor or np.ndarray)
        else:
            # 念のため transforms が None のとき用
            image = image.astype("float32") / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask)

        # マスクは (1, H, W) にしておく（1チャネル）
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)  # (1, H, W)

        return image, mask


# =========================
# 前処理・データ拡張
# =========================

def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


# =========================
# ユーティリティ
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model():
    """
    DeepLabV3Plus (encoder=resnet34, 2値セグメンテーション)
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


def calc_iou(preds, targets, threshold=0.5, eps=1e-6):
    """
    ミニバッチ単位のIoUを計算（2値マスク）
    preds  : (B,1,H,W) ロジット
    targets: (B,1,H,W) 0 or 1
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# =========================
# 学習ループ
# =========================

def train(
    images_dir="data/images",
    masks_dir="data/masks",
    batch_size=4,
    num_epochs=50,
    lr=1e-4,
    weight_decay=1e-4,
    val_ratio=0.2,
    num_workers=4,
    checkpoint_dir="checkpoints",
):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ==== データセット ====
    full_dataset = RoadDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transforms=get_train_transform(),
    )

    n_total = len(full_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    print(f"Total samples: {n_total}  (train: {n_train}, val: {n_val})")

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # val 用は transform を差し替え
    val_dataset.dataset.transforms = get_val_transform()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    # ==== モデル ====
    model = get_model().to(device)

    # 2値セグメンテーションなので BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_iou = 0.0

    for epoch in range(1, num_epochs + 1):
        # ----- train -----
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)         # (B,1,H,W) logits
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_iou += calc_iou(outputs.detach(), masks)

        train_loss /= len(train_loader.dataset)
        train_iou /= len(train_loader)

        # ----- val -----
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                val_iou += calc_iou(outputs, masks)

        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader)

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.4f}, train_iou={train_iou:.4f}, "
              f"val_loss={val_loss:.4f}, val_iou={val_iou:.4f}")

        # ベストモデル保存
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = os.path.join(checkpoint_dir, "deeplabv3plus_road_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> best model updated! saved to {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="data/images",
                        help="学習用画像ディレクトリ")
    parser.add_argument("--masks_dir", default="data/masks",
                        help="学習用マスクディレクトリ")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")

    args = parser.parse_args()

    train(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
    )
