预处理：
scripts/convert_dota_to_yolo.py   # 格式转换 + 类别映射
scripts/split_dota.py             # 大图滑动窗口分割
scripts/visualize_labels.py       # 验证可视化（可选）
训练：
train.py（已经训练好了）
评估与推理：
python validate.py
python predict.py         #预测推理
python predict_batch.py   #批量推理
python predict_large.py   # 针对超大幅遥感影像
python predict_webcam.py  # 实时检测
项目结构为：
├── data/                          # 数据集目录
│   ├── DOTA/                      # 原始DOTA数据集
│   │   ├── images/                # 原始图像
│   │   │   ├── train/             # 训练集图像
│   │   │   ├── val/               # 验证集图像
│   │   │   └── test/              # 测试集图像
│   │   └── labelTxt/              # 原始标注
│   │       ├── train/             # 训练集标注
│   │       ├── val/               # 验证集标注
│   │       └── test/              # 测试集标注
│   └── processed/                 # 预处理后的数据
│       ├── images/                # 分割后图像
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/                # 转换后标注
│           ├── train/
│           ├── val/
│           └── test/
├── scripts/                       # 脚本目录
│   ├── split_dota.py              # 图像分割脚本
│   ├── convert_dota_to_yolo.py    # 格式转换脚本
│   └── visualize_labels.py        # 标注可视化脚本
├── config/                        # 配置文件
│   └── dota.yaml                  # 数据集配置
├── checkpoints/                   # 模型权重保存目录
├── results/                       # 训练结果输出目录
│   ├── runs/                      # 训练日志
│   └── predictions/               # 预测结果
├── train.py                       # 训练主脚本
├── validate.py                    # 验证评估脚本
├── predict.py                     # 单图推理脚本
├── predict_large.py               # 大尺寸图像切片推理脚本
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明