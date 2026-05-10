#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量图像推理"""
import os
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
from predict import draw_obb_results   # 复用绘制函数

def batch_predict(model, input_dir, output_dir, conf=0.25, iou=0.45):
    os.makedirs(output_dir, exist_ok=True)
    exts = ['.jpg','.jpeg','.png','.tif','.tiff','.bmp']
    images = []
    for ext in exts:
        images.extend(list(Path(input_dir).glob(f'*{ext}')))
        images.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
    images = sorted(set(images))
    print(f"找到 {len(images)} 张图像")
    total_time = 0
    total_dets = 0
    for i, img_path in enumerate(images):
        start = time.time()
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = model(img, conf=conf, iou=iou)
        vis, dets = draw_obb_results(img, results, model.names, conf)
        out_path = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(out_path), vis)
        elapsed = time.time() - start
        total_time += elapsed
        total_dets += dets
        print(f"[{i+1}/{len(images)}] {img_path.name}: {dets} 个目标, 耗时 {elapsed:.2f}s")
    print(f"\n汇总: {len(images)} 张, 共 {total_dets} 目标, 平均 {total_time/len(images):.2f}s/张")

def main():
    base_dir = Path(__file__).parent
    model_path = base_dir / "runs" / "obb" / "exp" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = base_dir / "checkpoints" / "best.pt"
    input_dir = base_dir / "data" / "DOTA" / "images" / "test"
    output_dir = base_dir / "results" / "batch_predictions"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)
    batch_predict(model, input_dir, output_dir)

if __name__ == "__main__":
    main()