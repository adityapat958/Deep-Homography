# Deep Homography

A deep learning approach to estimate homography transformations between image pairs using supervised and unsupervised neural networks.

## Overview

This project implements a CNN-based system to predict homography parameters (8 coefficients) that align one image to another. It includes both supervised learning with ground-truth labels and unsupervised learning using photometric loss.

## Features

- **Supervised & Unsupervised Training**: Two learning paradigms for homography estimation
- **CNN Architecture**: 4-layer convolutional network with batch normalization and dropout
- **TensorBoard Logging**: Real-time visualization of training metrics
- **Flexible Dataset Support**: Works with custom image sets
- **Model Checkpointing**: Automatic saving of best models

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- OpenCV, NumPy, Kornia
- See `requirements.txt` for full dependencies

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
cd Phase2/Code
python Train.py  # Supervised training
```

### Testing
```bash
python Test.py   # Evaluate on test set
```

### Generate Visualizations
```bash
python generate_phase2_report.py      # Supervised results
python generate_phase2_report_unsupervised.py  # Unsupervised results
```

## Project Structure

```
Phase2/
├── Code/
│   ├── Train.py, Test.py              # Training & evaluation
│   ├── Network/Network.py              # CNN architecture
│   ├── Misc/                           # Utility functions
│   └── Logs/                           # TensorBoard logs
└── Data/
    ├── Train/, Val/                    # Training data
    ├── P1Ph2TestSet/                   # Test data
    └── results/                        # Output results
```

## Models

Pre-trained models available in `model_weights/`:
- `supervised_homography_net.pth` - Supervised trained model
- `unsupervised_homography_net.pth` - Unsupervised trained model

## References

Based on homography estimation from [DeTone et al., 2016] and photometric loss approaches for unsupervised learning.