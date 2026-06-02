#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滑动窗口分割 YOLO 格式标注的大尺寸遥感图像
输入：processed/images/ 和 processed/labels/（YOLO OBB格式）
输出：分割后的子图及对应的 YOLO 标注（仍保存在 processed/images/ 和 processed/labels/ 中，覆盖原文件）
注意：分割后原来的大图会被子图替换（或可另存目录），建议备份原 processed 目录
"""

import os
import cv2
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon


def clip_polygon_to_image(polygon, img_w, img_h):
    """裁剪多边形到图像边界内，返回顶点坐标列表（列表的列表）"""
    from shapely.geometry import Polygon as ShapelyPolygon
    img_polygon = ShapelyPolygon([(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)])
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


def split_image_single(img_path, label_path, save_dir_img, save_dir_label,
                       crop_size=(1024, 1020), overlap=(200, 200)):
    """
    对单张图像及其 YOLO OBB 标注进行滑动窗口分割
    Args:
        img_path: 图像路径（大图）
        label_path: 对应的 YOLO OBB 标注文件（每行 class_id x1 y1 x2 y2 x3 y3 x4 y4，坐标归一化）
        save_dir_img: 子图保存目录
        save_dir_label: 子图标注保存目录
        crop_size: (width, height)
        overlap: (x_overlap, y_overlap)
    Returns:
        (saved_count, total_objs)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图像: {img_path}")
        return 0, 0
    img_h, img_w = img.shape[:2]
    img_name = Path(img_path).stem

    # 读取 YOLO OBB 标注
    objs = []  # 每个元素为 (class_id, points_norm), points_norm 为 [(x1,y1),...,(x4,y4)] 归一化
    if not os.path.exists(label_path):
        print(f"警告: 标注文件不存在 {label_path}")
        return 0, 0
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            points_norm = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
            objs.append((class_id, points_norm))

    if len(objs) == 0:
        print(f"警告: {img_name} 没有有效标注")
        return 0, 0

    # 计算滑动窗口位置
    crop_w, crop_h = crop_size
    overlap_x, overlap_y = overlap
    step_x = crop_w - overlap_x
    step_y = crop_h - overlap_y
    num_windows_x = max(1, math.ceil((img_w - crop_w) / step_x) + 1)
    num_windows_y = max(1, math.ceil((img_h - crop_h) / step_y) + 1)

    windows = []
    for i in range(num_windows_x):
        for j in range(num_windows_y):
            x_start = min(i * step_x, img_w - crop_w)
            y_start = min(j * step_y, img_h - crop_h)
            x_end = min(x_start + crop_w, img_w)
            y_end = min(y_start + crop_h, img_h)
            windows.append((int(x_start), int(y_start), int(x_end), int(y_end)))

    saved_count = 0
    total_objs = 0

    for idx, (x1, y1, x2, y2) in enumerate(windows):
        window_w = x2 - x1
        window_h = y2 - y1
        if window_w < crop_w * 0.5 or window_h < crop_h * 0.5:
            continue
        crop_img = img[y1:y2, x1:x2]
        if crop_img.size == 0:
            continue

        window_labels = []  # 存储 [class_id, x1_norm, y1_norm, ...]

        for class_id, points_norm in objs:
            # 将归一化坐标转换为绝对坐标
            points_abs = [(p[0] * img_w, p[1] * img_h) for p in points_norm]
            # 构建多边形，并平移至子图坐标系
            poly_abs = Polygon(points_abs)
            poly_shifted = Polygon([(p[0] - x1, p[1] - y1) for p in points_abs])
            # 裁剪到子图边界
            clipped_polys = clip_polygon_to_image(poly_shifted, window_w, window_h)

            for clipped in clipped_polys:
                # 归一化到子图尺寸
                norm_points = []
                for px, py in clipped:
                    norm_x = px / window_w
                    norm_y = py / window_h
                    norm_points.extend([norm_x, norm_y])
                if len(norm_points) == 8:
                    # 确保坐标在 [0,1] 范围内
                    norm_points = [max(0.0, min(1.0, v)) for v in norm_points]
                    window_labels.append([class_id] + norm_points)

        if len(window_labels) == 0:
            continue

        # 保存子图
        out_img_path = os.path.join(save_dir_img, f"{img_name}_{idx:04d}.png")
        cv2.imwrite(out_img_path, crop_img)
        # 保存子图标注
        out_label_path = os.path.join(save_dir_label, f"{img_name}_{idx:04d}.txt")
        with open(out_label_path, 'w') as f:
            for label in window_labels:
                f.write(' '.join(map(str, label)) + '\n')
        saved_count += 1
        total_objs += len(window_labels)

    return saved_count, total_objs


def split_dataset(processed_root, split_names=None,
                  crop_size=(1024, 1020), overlap=(200, 200)):
    """
    批量分割整个预处理后的数据集（YOLO格式）
    Args:
        processed_root: data/processed 目录
        split_names: 要处理的子集列表，如 ['train', 'val', 'test']
        crop_size: 裁剪尺寸
        overlap: 重叠区域大小
    """
    if split_names is None:
        split_names = ['train', 'val', 'test']

    total_saved = 0
    total_objs = 0

    for split in split_names:
        print(f"\n正在处理 {split} 集...")
        img_dir = os.path.join(processed_root, 'images', split)
        label_dir = os.path.join(processed_root, 'labels', split)
        out_img_dir = img_dir   # 直接覆盖原目录（因为分割后大图不再需要，子图写入同一目录）
        out_label_dir = label_dir

        if not os.path.exists(img_dir):
            print(f"警告: {img_dir} 不存在，跳过")
            continue

        # 获取所有大图文件（png/jpg等）
        img_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        img_files = []
        for ext in img_extensions:
            img_files.extend(list(Path(img_dir).glob(f'*{ext}')))
            img_files.extend(list(Path(img_dir).glob(f'*{ext.upper()}')))

        # 注意：这里会遍历所有图像，分割后生成子图，原来的大图会被保留，不会自动删除。
        # 如果你想保留大图，建议先备份或分割到不同目录。这里为了简化，直接写入原目录。

        for img_path in tqdm(img_files, desc=f"分割{split}"):
            label_path = os.path.join(label_dir, f"{img_path.stem}.txt")
            if not os.path.exists(label_path):
                print(f"警告: {img_path.stem} 缺少标注文件，跳过")
                continue

            saved, objs = split_image_single(
                str(img_path), label_path,
                out_img_dir, out_label_dir,
                crop_size=crop_size, overlap=overlap
            )
            total_saved += saved
            total_objs += objs
            # 可选：分割完成后删除原大图和原标注（节省空间）
            # os.remove(img_path)
            # os.remove(label_path)

        print(f"{split}集: 原始图 {len(img_files)} 张，分割生成 {total_saved} 张子图，包含 {total_objs} 个标注")

    print(f"\n分割完成！共生成 {total_saved} 张子图，包含 {total_objs} 个标注目标")
    return total_saved, total_objs


def main():
    base_dir = Path(__file__).parent.parent
    processed_root = base_dir / "data" / "processed"
    split_dataset(processed_root,
                  split_names=['train', 'val', 'test'],
                  crop_size=(1024, 1020), overlap=(200, 200))


if __name__ == "__main__":
    main()