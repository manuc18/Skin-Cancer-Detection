# Project: Skin Cancer Detection
### ISIC 2024 — 3D-TBP Crops & Metadata Fusion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch Lightning](https://img.shields.io/badge/framework-PyTorch%20Lightning-purple.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem Statement

Population-scale skin cancer screening requires AI systems that are **accurate, calibrated, and clinically reliable**. Given a dermoscopic image from a 3D total body photography scan (3D-TBP) and patient metadata, predict whether a skin lesion is **malignant (1) or benign (0)**.

**Clinical constraint:** Missing a malignancy (false negative) is far more dangerous than a false alarm. The primary metric is **partial AUC (pAUC) at TPR ≥ 0.80**, forcing the model to maintain discriminative power at high sensitivity operating points.

---

## Dataset

- **Source:** [ISIC 2024 Challenge — SLICE-3D](https://www.kaggle.com/competitions/isic-2024-challenge)
- ~400,000 lesion crops from 3D total body photography
- ~3.5% positive rate (severe class imbalance)
- ~40 metadata features per lesion (age, sex, anatomical site, TBP-computed indices)

---

## Approach

We implement and compare four pathways in increasing complexity:

| Pathway | Description | Key Method |
|---------|-------------|------------|
| **Baseline ML** | Metadata-only risk prediction | LightGBM + feature engineering |
| **Advanced ML** | Classical CV features + metadata | HOG/LBP/Color → SVM/RF |
| **Deep Learning** | Image-only CNN classifier | EfficientNet-B4 + Focal Loss |
| **Hybrid Fusion** | Visual + clinical information | Image encoder + Metadata MLP + Calibration |

---

## Results

| Model | pAUC (TPR≥0.80) | AUC | Sensitivity | Specificity |
|-------|-----------------|-----|-------------|-------------|
| Baseline ML (LightGBM) | TBD | TBD | TBD | TBD |
| Advanced ML (HOG+SVM) | TBD | TBD | TBD | TBD |
| Deep Learning (EfficientNet) | TBD | TBD | TBD | TBD |
| Hybrid Fusion | TBD | TBD | TBD | TBD |

---

## Repository Structure

```
skin-cancer-detection/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline_ml.yaml        # LightGBM hyperparameters
│   ├── deep_learning.yaml      # EfficientNet training config
│   └── fusion.yaml             # Hybrid model config
├── data/
│   ├── raw/                    # ISIC 2024 downloads (gitignored)
│   └── processed/              # Cleaned, split datasets
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_baseline_ml.ipynb    # LightGBM baseline
│   ├── 03_advanced_ml.ipynb    # Classical CV features
│   ├── 04_deep_learning.ipynb  # EfficientNet training
│   └── 05_fusion.ipynb         # Hybrid model + calibration
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch datasets
│   │   ├── transforms.py       # Albumentations pipelines
│   │   └── preprocessing.py   # Feature engineering
│   ├── models/
│   │   ├── baseline.py         # LightGBM wrapper
│   │   ├── cnn.py              # EfficientNet/ViT encoder
│   │   ├── metadata_mlp.py     # Metadata branch
│   │   └── fusion.py           # Hybrid fusion model
│   ├── training/
│   │   ├── trainer.py          # PyTorch Lightning module
│   │   ├── losses.py           # Focal loss, weighted BCE
│   │   └── callbacks.py        # Early stopping, logging
│   └── evaluation/
│       ├── metrics.py          # pAUC, AUC, calibration
│       ├── calibration.py      # Temperature scaling
│       └── visualization.py   # ROC curves, reliability diagrams
├── experiments/                # W&B run logs (gitignored)
└── report/
    ├── main.tex                # LaTeX report
    └── figures/                # Generated plots
```

---

## Setup & Reproduction

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/skin-cancer-detection.git
cd skin-cancer-detection

# 2. Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download ISIC 2024 data
[ISIC 2024 Challenge — SLICE-3D](https://www.kaggle.com/competitions/isic-2024-challenge)

# 5. Run EDA
jupyter notebook notebooks/01_eda.ipynb

# 6. Train baseline model
python -m src.training.train_baseline --config configs/baseline_ml.yaml
```

---

## Key Design Decisions

- **GroupKFold by patient_id** — prevents data leakage (multiple lesions per patient)
- **Focal Loss** — addresses severe class imbalance (~3.5% positives) without oversampling
- **pAUC @ TPR≥0.80** — clinically meaningful metric, not raw accuracy
- **Temperature Scaling** — post-hoc calibration for reliable probability outputs
- **EfficientNet-B4** — strong ImageNet pretraining, efficient for medical imaging

---

## Team

- **Course:** Deep Learning & Advanced Machine Learning — Phase I
- **Members:** Manu Vahan (230125), Priyanshu Jangra (230157)

---

## References

1. Esteva et al. (2017). *Dermatologist-level classification of skin cancer with deep neural networks.* Nature.
2. Kurtansky et al. (2024). *The SLICE-3D dataset.* ISIC 2024.
3. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
4. Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
5. Codella et al. (2018). *Skin Lesion Analysis Toward Melanoma Detection.* ISIC 2018.
