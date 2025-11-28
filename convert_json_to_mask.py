#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON(annotation) → PNG(mask) 変換スクリプト（道路セグメンテーション用）

対応形式:
- BDD100K (frames[0].objects[].poly2d, category == "area/drivable")
- LabelMe
- DatasetNinja(仮想的な road polygon)
- COCO polygon

出力:
- mask は 0/255 の 1チャネル PNG
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- 各フォーマットごとの polygon 取得 ----------

def polygons_from_labelme(js):
    polys = []
    for shape in js["shapes"]:
        if shape["label"].lower() in ["road", "roads", "lane", "carriageway"]:
            pts = [(p[0], p[1]) for p in shape["points"]]
            polys.append(np.array(pts, np.int32))
    return polys


def polygons_from_datasetninja(js):
    polys = []
    anns = js.get("annotations", [])
    for ann in anns:
        if ann.get("classTitle", "").lower() == "road":
            pts = ann["points"]["exterior"]  # [[x,y], ...]
            polys.append(np.array(pts, np.int32))
    return polys


def polygons_from_coco(js):
    polys = []
    for ann in js["annotations"]:
        if ann.get("category_id") == 1:  # road と仮定
            segs = ann["segmentation"]
            for seg in segs:
                pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                polys.append(np.array(pts, np.int32))
    return polys


def polygons_from_bdd_drivable(js):
    """
    BDD100K の drivable area 用:
    - js["frames"][0]["objects"] の中から category == "area/drivable"
    - obj["poly2d"] は [x, y, "L" or "C"] のリスト
    """
    polys = []

    frames = js.get("frames", [])
    if not frames:
        return polys

    objects = frames[0].get("objects", [])

    for obj in objects:
        cat = obj.get("category", "")
        if cat != "area/drivable":
            continue
        poly2d = obj.get("poly2d", [])
        if not poly2d:
            continue

        pts = [(p[0], p[1]) for p in poly2d]  # 3番目の "L"/"C" は無視
        polys.append(np.array(pts, np.int32))

    return polys


# ---------- メイン変換処理 ----------

def guess_image_size_from_polys(polys, default=(720, 1280)):
    if not polys:
        return default
    xs = []
    ys = []
    for p in polys:
        xs.extend(p[:, 0].tolist())
        ys.extend(p[:, 1].tolist())
    w = int(max(xs)) + 2
    h = int(max(ys)) + 2
    return h, w


def find_image_path(image_dir: Path, name: str):
    """BDD100K など用: name から画像ファイルを探す"""
    if image_dir is None:
        return None
    candidates = [
        image_dir / f"{name}.jpg",
        image_dir / f"{name}.png",
        image_dir / f"{name}.jpeg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def convert(json_dir, out_mask_dir, image_dir=None):
    json_dir = Path(json_dir)
    out_mask_dir = Path(out_mask_dir)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(image_dir) if image_dir is not None else None

    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files.")

    for js_path in tqdm(json_files):
        js = load_json(js_path)

        polys = []
        h = w = None

        # ---- フォーマット判定 ----
        if "shapes" in js:  # LabelMe
            polys = polygons_from_labelme(js)
            h, w = js["imageHeight"], js["imageWidth"]

        elif "frames" in js:  # ★ BDD100K 形式
            polys = polygons_from_bdd_drivable(js)

            # 画像サイズは image_dir から取得するのが安全
            img_path = None
            if image_dir is not None:
                img_path = find_image_path(image_dir, js.get("name", ""))
                if img_path is not None:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]

            # 画像が見つからない場合はポリゴンから推定（だいたい 720x1280）
            if h is None or w is None:
                h, w = guess_image_size_from_polys(polys, default=(720, 1280))

        elif "annotations" in js and "images" in js:  # COCO
            polys = polygons_from_coco(js)
            img_info = js["images"][0]
            h, w = img_info["height"], img_info["width"]

        elif "annotations" in js:  # DatasetNinja を想定
            polys = polygons_from_datasetninja(js)
            if image_dir:
                img_path = find_image_path(image_dir, js["image"]["filename"])
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
            else:
                h, w = guess_image_size_from_polys(polys)

        else:
            print("Unsupported JSON format:", js_path)
            continue

        # ---- マスク生成 ----
        if h is None or w is None:
            print("Could not determine image size for", js_path)
            continue

        mask = np.zeros((h, w), dtype=np.uint8)

        for poly in polys:
            if poly.shape[0] >= 3:  # 少なくとも三角形
                cv2.fillPoly(mask, [poly], 255)

        out_name = js_path.stem + ".png"
        out_path = out_mask_dir / out_name
        cv2.imwrite(str(out_path), mask)

    print("Done! PNG masks saved to:", out_mask_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True, help="JSON annotation directory")
    parser.add_argument("--out_mask_dir", required=True, help="Output PNG mask directory")
    parser.add_argument("--image_dir", default=None, help="original images directory")
    args = parser.parse_args()

    convert(args.json_dir, args.out_mask_dir, args.image_dir)
