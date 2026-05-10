#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLOv8-OBB模型验证评估"""
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return None
    return YOLO(model_path)

def validate_model(model, data_yaml, imgsz=640, batch=16, conf=0.001, iou=0.6, device="0"):
    results = model.val(data=data_yaml, imgsz=imgsz, batch=batch,
                        conf=conf, iou=iou, device=device, workers=8,
                        split='val', plots=True, verbose=True)
    return results

def main():
    base_dir = Path(__file__).parent
    data_yaml = base_dir / "config" / "dota.yaml"
    model_path = base_dir / "runs" / "obb" / "exp" / "weights" / "best.pt"
    # 如果不存在，尝试从 checkpoints 目录读取

    device = "0" if torch.cuda.is_available() else "cpu"
    model = load_model(str(model_path))
    if model:
        validate_model(model, str(data_yaml), device=device)

if __name__ == "__main__":
    main()