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
- **Members:** Manu Vahan , Priyanshu Jangra 

