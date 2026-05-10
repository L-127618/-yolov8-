#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8-OBB模型训练脚本（显存优化版 + 损坏模型自动重试）
适用于 6GB 显存 GPU，使用 yolov8s-obb.pt 模型，batch=8
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def setup_training_environment():
    """设置训练环境"""
    print("=" * 60)
    print("环境配置")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    if cuda_available:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    device = "0" if cuda_available else "cpu"
    print(f"使用设备: {device}")

    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子: {seed}")

    return device


def create_model(model_name="yolov8s-obb.pt", num_classes=15):
    """
    创建YOLOv8-OBB模型，自动处理损坏的权重文件
    """
    local_path = Path(model_name)
    # 如果本地文件存在但大小异常（小于 1 MB），视为损坏并删除
    if local_path.exists() and local_path.stat().st_size < 1_000_000:
        print(f"警告: 检测到可能损坏的模型文件 {local_path} (大小 {local_path.stat().st_size} 字节)，正在删除...")
        local_path.unlink()
        print("已删除，将重新下载。")

    print(f"\n加载模型: {model_name}")
    model = YOLO(model_name)   # 如果文件不存在或已删除，会自动下载
    print(f"模型任务: {model.task}")
    return model


def train_model(model, data_yaml, epochs=100, imgsz=640, batch=8,
                device="0", lr0=0.01, patience=50):
    """
    训练模型（显存优化配置）
    """
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"数据集配置: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"图像尺寸: {imgsz}")
    print(f"批大小: {batch}")
    print(f"初始学习率: {lr0}")
    print(f"提前停止: {patience} epoch")

    results = model.train(
        data=data_yaml,          # 数据集配置文件
        epochs=epochs,          # 训练轮次
        imgsz=imgsz,            # 输入图像大小
        batch=batch,            # 批次大小（根据显存调整，6GB建议4或8）
        device=device,          # 使用的设备
        workers=4,              # 数据加载线程数（降低可减少内存占用）
        lr0=lr0,                # 初始学习率
        momentum=0.937,         # SGD动量
        weight_decay=0.0005,    # 权重衰减
        warmup_epochs=3,        # 预热轮数
        warmup_momentum=0.8,    # 预热动量
        warmup_bias_lr=0.1,     # 预热偏置学习率
        box=7.5,                # 盒损失增益
        cls=0.5,                # 分类损失增益
        dfl=1.5,                # DFL损失增益
        hsv_h=0.015,            # 色调增强
        hsv_s=0.7,              # 饱和度增强
        hsv_v=0.4,              # 明度增强
        degrees=0.0,            # 旋转角度
        translate=0.1,          # 平移增强
        scale=0.5,              # 缩放增强
        shear=0.0,              # 剪切增强
        perspective=0.0,        # 透视增强
        flipud=0.0,             # 上下翻转
        fliplr=0.5,             # 左右翻转
        mosaic=1.0,             # mosaic增强
        mixup=0.0,              # mixup增强
        copy_paste=0.0,         # copy-paste增强
        seed=42,                # 随机种子
        optimizer='auto',       # 自动选择优化器
        verbose=True,           # 详细输出
        save=True,              # 保存训练检查点
        save_period=-1,         # 每隔几个epoch保存
        val=True,               # 验证
        split='val',            # 验证集名称
        save_json=False,        # 是否保存JSON结果
        exist_ok=True,          # 已存在目录时覆盖
        resume=False,           # 断点续训
        project="runs/obb",     # 结果保存目录
        name="exp",             # 实验名称
        patience=patience,      # 早停耐心值
        plots=True,             # 生成训练曲线图
    )
    return results


def main():
    """主函数"""
    base_dir = Path(__file__).parent
    data_yaml = base_dir / "config" / "dota.yaml"

    if not data_yaml.exists():
        print(f"错误: 数据集配置文件不存在: {data_yaml}")
        print("请先创建 config/dota.yaml 文件")
        return

    # 读取类别数量
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
        num_classes = data_config.get('nc', 15)
        print(f"\n数据集信息:")
        print(f"  类别数量: {num_classes}")
        print(f"  数据集路径: {data_config.get('path', 'N/A')}")

    # 环境设置
    device = setup_training_environment()

    # 选择模型（显存有限建议使用 yolov8s-obb.pt）
    MODEL_NAME = "yolov8s-obb.pt"   # 11.4M 参数，适合 6GB 显存
    # 其他可选: yolov8n-obb.pt (更小), yolov8m-obb.pt (需减小 batch)

    model = create_model(MODEL_NAME, num_classes)

    # 开始训练（batch 根据显存调整，6GB 建议 8 或 4）
    train_model(
        model=model,
        data_yaml=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=8,          # 如果仍 OOM 改为 4
        device=device,
        lr0=0.01,
        patience=50
    )

    print("\n训练完成！")
    print(f"最佳模型保存在: runs/obb/exp/weights/best.pt")
    print(f"训练日志: runs/obb/exp")


if __name__ == "__main__":
    main()