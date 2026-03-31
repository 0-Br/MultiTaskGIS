# MultiTaskGIS

## 简介 / Introduction

基于掩码自编码器（Masked Autoencoders, MAE）的多任务地理信息系统模型，面向遥感影像语义分割任务。本项目提出了一种基于迁移学习的多任务 GIS 范式：以预训练的 ViT-MAE 作为视觉编码器（冻结参数），UNet 作为分割解码器，在低数据需求和易部署的条件下实现了优异的分割性能。对比实验表明，该方法在语义分割任务上优于传统遥感影像处理范式。

A multi-task geospatial information system model based on Masked Autoencoders (MAE) for remote sensing image semantic segmentation. This project proposes a transfer learning-based multi-task GIS paradigm: by leveraging a pretrained ViT-MAE as the vision encoder (frozen weights) and a UNet as the segmentation decoder, the model achieves strong performance with low data requirements and easy deployment. Comparative experiments demonstrate the superiority of this approach over traditional remote sensing image processing paradigms.

## 模型架构 / Architecture

本项目提供两种模型配置：

Two model configurations are available:

- **UNet**：标准 CNN 编码器-解码器语义分割模型（基线）/ Standard CNN-based encoder-decoder for semantic segmentation (baseline)
- **MAEtoUNet**：冻结的 ViT-MAE 编码器（ImageNet 预训练，`facebook/vit-mae-base`）+ UNet 解码器——本项目提出的迁移学习方法 / Frozen Vision Transformer MAE encoder (pretrained on ImageNet, `facebook/vit-mae-base`) + UNet decoder — the proposed transfer learning approach

## 数据集 / Dataset

本项目使用 [LoveDA](https://github.com/Junjue-Wang/LoveDA) 城市子集，包含 8 类地物覆盖：背景、建筑、道路、水体、裸地、森林、农田、无数据。

This project uses the [LoveDA](https://github.com/Junjue-Wang/LoveDA) urban subset with 8 land cover classes: background, building, road, water, barren, forest, agriculture, and no-data.

### 数据准备 / Data Preparation

1. 下载 LoveDA 数据集，放置到本地路径（如 `./2021LoveDA/`）
2. 运行 `data/preprocess.ipynb`，将原始影像裁剪并转换为 224×224 的 `.npy` 格式；根据需要修改 notebook 中的 `SOURCE_DIR` 和 `OUTPUT_DIR`
3. 更新 `train.yaml` 中的 `data.DB_dir` 字段，指向处理后的数据目录

1. Download the LoveDA dataset and place it at a local path (e.g., `./2021LoveDA/`)
2. Run `data/preprocess.ipynb` to crop and convert images into 224×224 `.npy` patches; update `SOURCE_DIR` and `OUTPUT_DIR` in the notebook as needed
3. Update the `data.DB_dir` field in `train.yaml` to point to the processed output directory

## 预训练权重 / Pretrained Weights

MAE 编码器使用 `facebook/vit-mae-base` 预训练权重（约 428MB，未包含在仓库中）。配置方法如下：

The MAE encoder uses the `facebook/vit-mae-base` pretrained weights (~428MB, not included in the repository). To set up:

```bash
mkdir -p models/pretrained
# 从 https://huggingface.co/facebook/vit-mae-base 下载以下文件
# Download the following files from https://huggingface.co/facebook/vit-mae-base
# 将 config.json、preprocessor_config.json 和 pytorch_model.bin 放入 models/pretrained/
# Place config.json, preprocessor_config.json, and pytorch_model.bin into models/pretrained/
```

## 训练 / Training

```bash
# 使用 MAEtoUNet 训练（默认） / Train with MAEtoUNet (default)
python train.py --config train.yaml --model MAEtoUNet

# 使用 UNet 训练 / Train with plain UNet
python train.py --config train.yaml --model UNet
```

训练超参数在 `train.yaml` 中配置，采用 AdamW 优化器、带预热的余弦学习率调度和早停策略。

Training hyperparameters are configured in `train.yaml`, using AdamW optimizer, cosine learning rate schedule with warmup, and early stopping.

## 项目结构 / Project Structure

```
├── train.py              # 训练入口 / Training entry point (PyTorch Lightning)
├── train.yaml            # 训练配置 / Training configuration
├── models/
│   ├── MAE.py            # ViT-MAE 编码器与 MAEtoUNet 桥接 / ViT-MAE encoder and MAEtoUNet bridge
│   ├── UNet.py           # UNet 分割解码器 / UNet segmentation decoder
│   └── configs/          # 模型架构配置 / Model architecture configs (JSON)
├── data/
│   ├── dataset.py        # RSDataset 数据加载 / RSDataset loader
│   └── preprocess.ipynb  # 原始数据预处理 / Raw data → .npy preprocessing
└── utils/
    ├── learn.py          # 学习率调度与实验工具 / LR scheduler and experiment utilities
    └── metrics.py        # 分割评估指标 / Segmentation metrics (accuracy, F1)
```

## 依赖 / Dependencies

- PyTorch & PyTorch Lightning
- HuggingFace Transformers
- OmegaConf
- scikit-learn
- NumPy, torchvision, Pillow

## 作者 / Author

刘滨瑞，清华大学
Binrui Liu, Tsinghua University
📧 lbr21@mails.tsinghua.edu.cn
