#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大尺寸遥感图像切片辅助推理（SAHI）"""
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def load_model_with_sahi(model_path, device="cuda:0", confidence_threshold=0.25):
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return None
    if device == "cuda:0" and not torch.cuda.is_available():
        device = "cpu"
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', model_path=model_path,
        confidence_threshold=confidence_threshold, device=device
    )
    return detection_model

def main():
    base_dir = Path(__file__).parent
    model_path = base_dir / "runs" / "dota" / "exp" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = base_dir / "checkpoints" / "best.pt"
    input_image = base_dir / "data" / "DOTA" / "images" / "test" / "large_example.tif"
    output_dir = base_dir / "results" / "sahi_predictions"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model_with_sahi(str(model_path), device, confidence_threshold=0.25)
    if model is None:
        return
    result = get_sliced_prediction(
        image=str(input_image), detection_model=model,
        slice_height=1024, slice_width=1024,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_match_threshold=0.5, postprocess_match_metric="IOU"
    )
    result.export_visuals(export_dir=str(output_dir), file_name=f"{input_image.stem}_sahi_detection.png")
    print(f"检测完成，共 {len(result.object_prediction_list)} 个目标，结果保存在 {output_dir}")

if __name__ == "__main__":
    main()