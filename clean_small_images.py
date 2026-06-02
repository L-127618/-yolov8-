#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理分割后尺寸过小的异常子图（宽或高 < min_size 像素）
同时删除对应的标注文件，避免训练时 ignored corrupt image 警告
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm


def clean_small_images(data_root, min_size=32, splits=None):
    """
    删除过小的图像及对应标注文件
    Args:
        data_root: processed 目录路径
        min_size: 最小边长（像素），小于该值则删除
        splits: 要处理的子集列表，如 ['train', 'val']
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    removed_count = 0
    for split in splits:
        img_dir = Path(data_root) / 'images' / split
        if not img_dir.exists():
            continue
        print(f"\n清理 {split} 集...")

        for img_path in tqdm(list(img_dir.glob('*.*'))):
            # 跳过常见非图像文件（谨慎）
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                continue
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  无效图像（无法读取）: {img_path}")
                    os.remove(img_path)
                    label_path = img_path.with_suffix('.txt')
                    if label_path.exists():
                        os.remove(label_path)
                    removed_count += 1
                    continue
                h, w = img.shape[:2]
                if h < min_size or w < min_size:
                    print(f"  删除过小图像: {img_path} ({w}x{h})")
                    os.remove(img_path)
                    label_path = img_path.with_suffix('.txt')
                    if label_path.exists():
                        os.remove(label_path)
                    removed_count += 1
            except Exception as e:
                print(f"  处理 {img_path} 时出错: {e}")
    print(f"\n清理完成，共删除 {removed_count} 个异常图像及对应标注。")


def main():
    base_dir = Path(__file__).parent.parent
    processed_root = base_dir / "data" / "processed"
    clean_small_images(processed_root, min_size=32, splits=['train', 'val'])


if __name__ == "__main__":
    main()