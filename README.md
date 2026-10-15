# Cell Segmentation with U-Net

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![CI](https://github.com/Helios337/Cell-Segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/Helios337/Cell-Segmentation/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automatically detect, segment, and count nuclei in fluorescence microscopy images using a **U-Net** deep learning model with ResNet50 pretrained encoder and watershed-based separation of overlapping cells.

## Key Features

- **U-Net with ResNet50 Encoder** - Pretrained on ImageNet for strong feature extraction
- **Two-Phase Training** - Frozen encoder first, then full fine-tuning
- **Focal Loss** - Handles extreme class imbalance in cell segmentation
- **Watershed Post-Processing** - Separates touching/overlapping nuclei for accurate counting
- **Test-Time Augmentation** - Averages predictions across flips for robust results
- **BBBC Dataset Support** - Download real microscopy data from the Broad Bioimage Benchmark Collection
- **Post-Processing Optimization** - Grid search over threshold and min_size for best cell count accuracy
- **Comprehensive Evaluation** - IoU, Dice, Precision, Recall, F1, and Cell Count MAE
- **Google Drive Checkpointing** - Save models persistently on Colab

## Quick Start

### Local Setup

```bash
git clone https://github.com/Helios337/Cell-Segmentation.git
cd Cell-Segmentation
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Train on BBBC038 (nuclei segmentation)
python main.py --data-source BBBC038
```

### Colab Setup

1. Open a new Colab notebook and connect to a GPU runtime (Runtime → Change runtime type → GPU).
2. Run the setup script:

```bash
!git clone https://github.com/Helios337/Cell-Segmentation.git
%cd Cell-Segmentation
!python colab_setup.py
```

Or set up manually:

```bash
!git clone https://github.com/Helios337/Cell-Segmentation.git
%cd Cell-Segmentation
!pip install -e .
!python main.py --mode train --data-source BBBC038 --epochs-phase1 10 --epochs-phase2 20
```

## CLI Usage

```bash
# Train on BBBC038 nuclei data (default mode)
python main.py --mode train --data-source BBBC038

# Train with custom hyperparameters
python main.py --data-source BBBC038 --epochs-phase1 10 --epochs-phase2 20 \
  --batch-size 8 --lr-phase1 0.001 --lr-phase2 0.00001

# Quick demo on synthetic data (no download needed)
python main.py --data-source synthetic --n-samples 50 --epochs-phase1 1 --epochs-phase2 1

# Evaluate a trained model on the test split
python main.py --mode eval --data-source BBBC038 --model-path best_model.keras

# Predict on a single image
python main.py --mode predict --image-path /path/to/cell.png --model-path best_model.keras

# With test-time augmentation
python main.py --data-source BBBC038 --tta

# Optimize post-processing thresholds
python main.py --data-source BBBC038 --optimize-thresholds
```

All hyperparameters default to the values in `config.yaml` and can be overridden on the command line.

## Project Structure

```
├── main.py                 # CLI pipeline runner (train / eval / predict)
├── model.py                # U-Net + ResNet50 encoder + training + evaluation
├── data_handler.py         # BBBC downloader + real/synthetic data loader
├── utils.py                # Augmentation, config loading, CSV export
├── config.yaml             # Default hyperparameters (loaded by main.py)
├── colab_setup.py          # One-shot Colab setup + training helper
├── tests/                  # Unit tests (model, data handler, utils)
├── pyproject.toml          # Package metadata and build config
├── requirements.txt        # Runtime dependencies
├── Makefile                # Common commands (install, train, test, lint)
├── Dockerfile              # Containerized deployment
└── .github/workflows/ci.yml
```

## Architecture

The U-Net follows the original Ronneberger et al. design with a pretrained ResNet50 encoder:

- **Encoder**: ResNet50 pretrained on ImageNet (conv1_relu → conv5_block3_out)
- **Bottleneck**: ResNet50 final feature map (2048 channels)
- **Decoder**: 4 blocks of Conv2DTranspose → Concatenate (skip) → Conv2D → Dropout → Conv2D
- **Output**: 1×1 Conv2D with sigmoid activation

## Loss Function

Combined BCE + Dice + Focal Loss:

`L = BCE(y, ŷ) + (1 - Dice(y, ŷ)) + Focal(y, ŷ)`

This handles class imbalance (nuclei occupy a small fraction of the image) better than any single loss.

## Training Strategy

1. **Phase 1** (frozen encoder): Train decoder only for 10 epochs with lr=1e-3
2. **Phase 2** (fine-tune): Unfreeze encoder, train entire model for 20 epochs with lr=1e-5
3. **Early stopping** with patience=10 on validation loss
4. **ReduceLROnPlateau** with factor=0.5, patience=5

## Evaluation Metrics

| Metric | Description |
|---|---|
| IoU (Jaccard) | Intersection over Union |
| Dice Coefficient | F1 score for segmentation overlap |
| Precision | False positive rate |
| Recall | False negative rate |
| F1 | Harmonic mean of precision and recall |
| Count MAE | Mean absolute error in cell count |

## Results

On BBBC038 (Kaggle 2018 Data Science Bowl):

| Metric | Value |
|---|---|
| IoU | ~0.75–0.85 |
| Dice Coefficient | ~0.85–0.92 |
| Count MAE | ±1–3 cells |

## Test

```bash
make test
# or
python -m pytest tests/ -v
```

## License

MIT