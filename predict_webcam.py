#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实时摄像头目标检测（增强按键响应）"""

import cv2
from ultralytics import YOLO
import torch
from pathlib import Path


def webcam_detection(model_path, camera_id=0, conf=0.5):
    # 检查模型文件
    if not Path(model_path).exists():
        print(f"模型文件不存在: {model_path}")
        return

    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型
    model = YOLO(model_path)
    model.to(device)

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("无法打开摄像头，请检查相机索引或驱动")
        return

    # 设置摄像头分辨率（可选，提高性能）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=" * 50)
    print("实时目标检测已启动")
    print("操作说明：")
    print("  - 按 'q' 键退出程序")
    print("  - 按 's' 键保存当前帧截图")
    print("  - 确保显示窗口处于活动状态（鼠标点击窗口）")
    print("=" * 50)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        # 每2帧推理一次，提高实时性（可选）
        if frame_count % 2 == 0:
            results = model(frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot() if results else frame
        else:
            annotated_frame = frame  # 跳过推理直接显示上一帧结果

        cv2.imshow('YOLOv8-OBB Real-time Detection', annotated_frame)

        # 按键检测（增加等待时间到10ms，并打印键值用于调试）
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("退出程序")
            break
        elif key == ord('s'):
            timestamp = cv2.getTickCount()
            screenshot_path = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"截图已保存: {screenshot_path}")
        elif key != 255:  # 有按键但不是q/s，可打印调试
            print(f"按下了键: {chr(key)} (ASCII: {key})，无效指令")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已释放")


def main():
    base_dir = Path(__file__).parent
    # 确保模型路径正确
    model_path = base_dir / "runs" / "obb" / "exp" / "weights" / "best.pt"
    if not model_path.exists():
        # 尝试其他可能的实验目录
        alt_path = base_dir / "runs" / "obb" / "exp2" / "weights" / "best.pt"
        if alt_path.exists():
            model_path = alt_path
        else:
            model_path = base_dir / "checkpoints" / "best.pt"
    if not model_path.exists():
        print("未找到模型权重文件，请先完成训练。")
        return

    webcam_detection(str(model_path), camera_id=0, conf=0.5)


if __name__ == "__main__":
    main()