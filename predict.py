#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单张图像推理（交互式输入图片路径，支持拖拽）"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch


def load_model(model_path):
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        return None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    return YOLO(str(model_path))


def draw_obb_results(image, results, class_names=None, conf_threshold=0.25):
    img_copy = image.copy()
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)
    ]
    total = 0
    for result in results:
        if not hasattr(result, 'obb') or result.obb is None:
            continue
        obb = result.obb
        corners = obb.xyxyxyxy.cpu().numpy()  # (n,4,2)
        cls_ids = obb.cls.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        for corner_arr, cls_id, conf in zip(corners, cls_ids, confs):
            if conf < conf_threshold:
                continue
            points = [(int(corner_arr[i][0]), int(corner_arr[i][1])) for i in range(4)]
            pts = np.array(points, dtype=np.int32)
            color = colors[int(cls_id) % len(colors)]
            cv2.polylines(img_copy, [pts], True, color, 2)
            if class_names and int(cls_id) < len(class_names):
                label = f"{class_names[int(cls_id)]} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                x0, y0 = points[0]
                cv2.rectangle(img_copy, (x0, y0 - th - 4), (x0 + tw, y0), color, -1)
                cv2.putText(img_copy, label, (x0, y0 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            total += 1
    return img_copy, total


def main():
    base_dir = Path(__file__).parent

    # 模型路径（根据你的训练结果调整）
    model_path = base_dir / "runs" / "obb" / "exp" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = base_dir / "runs" / "obb" / "exp2" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = base_dir / "checkpoints" / "best.pt"
    if not model_path.exists():
        print("❌ 未找到模型权重文件，请确认训练已完成。")
        return

    # 交互式输入图片路径
    print("=" * 60)
    print("请将图片文件拖拽到此窗口，然后按回车")
    print("（或者手动输入完整的图片路径）")
    print("=" * 60)
    img_path_str = input().strip().strip('"')   # 去除两端空格和引号
    img_path = Path(img_path_str)

    if not img_path.exists():
        print(f"❌ 图片不存在: {img_path}")
        return

    output_dir = base_dir / "results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    if model is None:
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print("❌ 无法读取图像，请确保文件是图片格式（jpg/png/tif等）")
        return

    print(f"🔍 正在推理: {img_path.name}")
    results = model(img, conf=0.25, iou=0.45)
    vis_img, num = draw_obb_results(img, results, model.names, 0.25)

    out_path = output_dir / f"detected_{img_path.name}"
    cv2.imwrite(str(out_path), vis_img)
    print(f"✅ 检测到 {num} 个目标，结果保存至: {out_path}")

    # 打印详细检测信息
    if results[0].obb is not None:
        obb = results[0].obb
        cls_ids = obb.cls.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
            if conf >= 0.25:
                class_name = model.names[int(cls_id)]
                print(f"   {i+1}. {class_name}: {conf:.4f}")


if __name__ == "__main__":
    main()