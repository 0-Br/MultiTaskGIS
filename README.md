# MultiTaskGIS

A multi-task geospatial information system model based on Masked Autoencoders (MAE) for remote sensing image semantic segmentation.

This project proposes a new multi-task GIS paradigm using transfer learning. By leveraging a pretrained MAE as the vision encoder and a UNet as the segmentation decoder, the model achieves strong performance with low data requirements and easy deployment. Comparative experiments on the segmentation task demonstrate the superiority of this approach over traditional remote sensing image processing paradigms.

## Architecture

Two model configurations are available:

- **UNet**: Standard CNN-based encoder-decoder for semantic segmentation
- **MAEtoUNet**: Vision Transformer MAE encoder (frozen, pretrained on ImageNet) + UNet decoder — the proposed transfer learning approach

## Dataset

This project uses the [LoveDA](https://github.com/Junjue-Wang/LoveDA) urban subset with 8 land cover classes: background, building, road, water, barren, forest, agriculture, and no-data.

### Data Preparation

1. Download the LoveDA dataset and place it at a local path (e.g., `./2021LoveDA/`).
2. Run `data/preprocess.ipynb` to crop and convert the images into 224×224 `.npy` patches. Update `SOURCE_DIR` and `OUTPUT_DIR` in the notebook as needed.
3. Update the `data.DB_dir` field in `train.yaml` to point to the processed output directory.

## Pretrained Weights

The MAE encoder uses the `facebook/vit-mae-base` pretrained weights. To set up:

```bash
mkdir -p models/pretrained
# Download from https://huggingface.co/facebook/vit-mae-base
# Place config.json, preprocessor_config.json, and pytorch_model.bin into models/pretrained/
```

## Training

```bash
# Train with MAEtoUNet (default)
python train.py --config train.yaml --model MAEtoUNet

# Train with plain UNet
python train.py --config train.yaml --model UNet
```

Training hyperparameters are configured in `train.yaml`.

## Project Structure

```
├── train.py              # Training entry point (PyTorch Lightning)
├── train.yaml            # Training configuration
├── models/
│   ├── MAE.py            # ViT-MAE encoder and MAEtoUNet bridge
│   ├── UNet.py           # UNet segmentation decoder
│   └── configs/          # Model architecture configs (JSON)
├── data/
│   ├── dataset.py        # RSDataset loader
│   └── preprocess.ipynb  # Raw data → .npy preprocessing
└── utils/
    ├── learn.py          # LR scheduler and experiment utilities
    └── metrics.py        # Segmentation metrics (accuracy, F1)
```

## Dependencies

- PyTorch & PyTorch Lightning
- HuggingFace Transformers
- OmegaConf
- scikit-learn
- NumPy, torchvision, Pillow
