# WWR Prediction Model / WWR预测模型

基于街景图像语义分割的建筑窗墙比（Window-to-Wall Ratio, WWR）预测模型。仓库记录 M1–M5 五个迭代版本，包含统一训练入口、模型结构、配置文件和迭代说明。

> 当前论文最终采用 **M4**；M5 为预训练编码器与高分辨率输入的探索性对照。

## 模型版本

| 版本 | 结构与目标 | 样本数 | 输入 | 最佳轮次 | 验证损失 |
|---|---|---:|---:|---:|---:|
| M1 | 单阶段三分类 U-Net | 101 | 512×512 | 54 | 0.4225 |
| M2 | 样本扩充的单阶段三分类 U-Net | 248 | 512×512 | 45 | 0.3095 |
| M3 | 建筑/背景 + 墙体/开口两阶段 U-Net | 248 | 512×512 | 44 / 25 | 0.1270 / 0.2936 |
| **M4** | **引入道路地面困难负样本的两阶段 U-Net** | **248** | **512×512** | **48 / 28** | **0.2397 / 0.2878** |
| M5 | ImageNet 预训练 ResNet18-U-Net | 300 | 1024×1024 | 57 | 0.2792 |

详细问题、改动和选型依据见 [`docs/model-iteration.md`](docs/model-iteration.md)。

## WWR 定义

```text
WWR = opening_pixels / (wall_pixels + opening_pixels)
```

开口类包含窗和门。道路、地面、天空、植被和车辆等背景像素不进入分母。

## 数据格式

数据采用 Labelme 形式，图像与 JSON 标注同名：

```text
data/
├── image001.png
├── image001.json
├── image002.png
└── image002.json
```

支持的标签为 `building`、`window`、`door`、`road` 和 `ground`。仓库只提供虚构标注格式示例，不包含原始街景图像、人工标注数据或历史预测图。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 训练单阶段版本
python train.py --config configs/m2.json --data-dir /path/to/data

# 训练两阶段 M4
python train.py --config configs/m4.json --data-dir /path/to/data --stage building
python train.py --config configs/m4.json --data-dir /path/to/data --stage opening

# 训练 M5（首次执行会下载 torchvision 预训练权重）
python train.py --config configs/m5.json --data-dir /path/to/data
```

每个配置默认按验证损失保存唯一最佳权重，避免将不同轮次的单项最优指标组合为不存在的模型。

## 数据增强与损失

- 水平翻转、小角度旋转、亮度/对比度扰动和高斯噪声。
- 损失函数为 `0.5 × CrossEntropy + 0.5 × macro Dice`。
- 随机种子为 42，默认训练/验证比例为 80%/20%。
- 优化器为 Adam；M5 对预训练编码器使用更小的学习率。

## 安全与数据边界

- 不包含 API 密钥、访问令牌或 `.env` 文件。
- 不包含百度街景原图、处理图、标注数据和预测结果。
- 不提交模型权重；`.pth`/`.pt` 已被 `.gitignore` 排除。
- 请确保使用的街景数据符合数据来源的授权条款。

## License

MIT

