#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DOTA数据集格式转YOLO OBB格式（修复版）
"""

import os
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from shapely.geometry import Polygon
warnings.filterwarnings('ignore')


def clip_polygon_to_image(polygon, img_w, img_h):
    """
    将任意多边形裁剪到图像边界内，返回一个或多个裁剪后的多边形（顶点坐标列表）
    """
    img_polygon = Polygon([(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    clipped = polygon.intersection(img_polygon)
    if clipped.is_empty:
        return []

    result = []
    if clipped.geom_type == 'Polygon':
        coords = list(clipped.exterior.coords)[:-1]
        if len(coords) >= 4:
            result.append(coords)
    elif clipped.geom_type == 'MultiPolygon':
        for poly in clipped.geoms:
            coords = list(poly.exterior.coords)[:-1]
            if len(coords) >= 4:
                result.append(coords)
    return result


def polygon_to_obb_points(poly, img_w, img_h):
    """将多边形四点坐标归一化为0~1范围"""
    if len(poly) != 8:
        raise ValueError(f"多边形需要8个坐标，实际为: {len(poly)}")
    normalized = []
    for i in range(0, 8, 2):
        x = max(0, min(poly[i], img_w))
        y = max(0, min(poly[i+1], img_h))
        normalized.append(x / img_w)
        normalized.append(y / img_h)
    # 确保归一化值在[0,1]
    return [max(0.0, min(1.0, v)) for v in normalized]


def convert_dota_to_yolo_obb(dota_root, output_root, class_mapping=None):
    """转换函数，路径使用相对项目根目录的写法"""
    if class_mapping is None:
        class_mapping = {
            'plane': 0, 'ship': 1, 'storage-tank': 2, 'baseball-diamond': 3,
            'tennis-court': 4, 'basketball-court': 5, 'ground-track-field': 6,
            'harbor': 7, 'bridge': 8, 'large-vehicle': 9, 'small-vehicle': 10,
            'helicopter': 11, 'roundabout': 12, 'soccer-ball-field': 13, 'swimming-pool': 14
        }
    splits = ['train', 'val']
    if os.path.exists(os.path.join(dota_root, 'labelTxt', 'test')):
        splits.append('test')

    for split in splits:
        print(f"\n处理 {split} 集...")
        img_dir = os.path.join(dota_root, 'images', split)
        label_dir = os.path.join(dota_root, 'labelTxt', split)
        out_img_dir = os.path.join(output_root, 'images', split)
        out_label_dir = os.path.join(output_root, 'labels', split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        if not os.path.exists(img_dir):
            print(f"警告: {img_dir} 不存在，跳过")
            continue

        img_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        img_files = []
        for ext in img_extensions:
            img_files.extend(list(Path(img_dir).glob(f'*{ext}')))
            img_files.extend(list(Path(img_dir).glob(f'*{ext.upper()}')))

        for img_path in tqdm(img_files, desc=f"转换{split}"):
            img_name = img_path.stem
            label_file = os.path.join(label_dir, f"{img_name}.txt")
            if not os.path.exists(label_file):
                print(f"警告: {img_name} 没有对应的标注文件，跳过")
                continue

            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图像 {img_path}，跳过")
                continue
            img_h, img_w = img.shape[:2]

            # 复制原始图像（不改变原图尺寸）
            import shutil
            shutil.copy2(str(img_path), os.path.join(out_img_dir, img_path.name))

            yolo_labels = []
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('imagesource') or line.startswith('gsd'):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                coords = list(map(float, parts[:8]))
                class_name = parts[8]
                if class_name not in class_mapping:
                    alt_name = class_name.replace('_', '-')
                    if alt_name in class_mapping:
                        class_name = alt_name
                    else:
                        print(f"警告: 未知类别 {class_name}，跳过")
                        continue
                class_id = class_mapping[class_name]

                try:
                    normalized_coords = polygon_to_obb_points(coords, img_w, img_h)
                    yolo_labels.append([class_id] + normalized_coords)
                except ValueError as e:
                    print(f"警告: {img_name} 中多边形转换失败: {e}")

            if yolo_labels:
                output_label_file = os.path.join(out_label_dir, f"{img_name}.txt")
                with open(output_label_file, 'w') as f:
                    for label in yolo_labels:
                        f.write(' '.join(map(str, label)) + '\n')
            else:
                print(f"警告: {img_name} 转换后无有效标注")

        print(f"{split}集转换完成: {len(img_files)} 张图像")
    print(f"\n转换完成！输出目录: {output_root}")
    return class_mapping


def main():
    # 项目根目录为 satellite_detection/
    base_dir = Path(__file__).parent.parent  # 从 scripts/ 向上两级
    DOTA_ROOT = base_dir / "data" / "DOTA"
    OUTPUT_ROOT = base_dir / "data" / "processed"
    class_mapping = convert_dota_to_yolo_obb(str(DOTA_ROOT), str(OUTPUT_ROOT))
    print("\n类别映射:", class_mapping)
    # 保存映射
    import json
    with open(OUTPUT_ROOT / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)


if __name__ == "__main__":
    main()