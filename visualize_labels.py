#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO OBB标注可视化工具"""
import os
import cv2
import numpy as np
from pathlib import Path


def draw_obb_on_image(image, obb_labels, class_names=None):
    img_h, img_w = image.shape[:2]
    img_copy = image.copy()
    colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0),
              (128,0,128),(255,128,0),(0,128,255),(128,255,0)]
    for label in obb_labels:
        if len(label) != 9:
            continue
        class_id = int(label[0])
        points = []
        for i in range(1, 9, 2):
            x = label[i] * img_w
            y = label[i+1] * img_h
            points.append([int(x), int(y)])
        pts = np.array(points, dtype=np.int32)
        color = colors[class_id % len(colors)]
        cv2.polylines(img_copy, [pts], True, color, 2)
        cx = int(np.mean([p[0] for p in points]))
        cy = int(np.mean([p[1] for p in points]))
        cv2.circle(img_copy, (cx, cy), 3, color, -1)
        if class_names and class_id < len(class_names):
            label_text = class_names[class_id]
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label_text, font, 0.5, 1)
            cv2.rectangle(img_copy, (points[0][0], points[0][1]-th-4),
                          (points[0][0]+tw, points[0][1]), color, -1)
            cv2.putText(img_copy, label_text, (points[0][0], points[0][1]-2),
                        font, 0.5, (255,255,255), 1)
    return img_copy


def visualize_dataset(image_dir, label_dir, output_dir, class_names=None, max_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    images = list(Path(image_dir).glob('*.png')) + list(Path(image_dir).glob('*.jpg'))
    print(f"找到 {len(images)} 张图像，将可视化前 {min(max_samples, len(images))} 张")
    count = 0
    for img_path in images[:max_samples]:
        label_path = os.path.join(label_dir, f"{img_path.stem}.txt")
        if not os.path.exists(label_path):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 9:
                    labels.append(parts)
        vis = draw_obb_on_image(img, labels, class_names)
        out_path = os.path.join(output_dir, f"{img_path.stem}_viz.jpg")
        cv2.imwrite(out_path, vis)
        count += 1
    print(f"可视化完成，共处理 {count} 张图像")


def main():
    base_dir = Path(__file__).parent.parent
    processed_root = base_dir / "data" / "processed"
    vis_output = base_dir / "results" / "visualizations"
    class_names = ["plane","ship","storage-tank","baseball-diamond","tennis-court",
                   "basketball-court","ground-track-field","harbor","bridge",
                   "large-vehicle","small-vehicle","helicopter","roundabout",
                   "soccer-ball-field","swimming-pool"]
    train_img = processed_root / "images" / "train"
    train_label = processed_root / "labels" / "train"
    visualize_dataset(str(train_img), str(train_label), str(vis_output / "train"),
                      class_names, max_samples=20)


if __name__ == "__main__":
    main()