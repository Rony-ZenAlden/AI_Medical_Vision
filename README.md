# AI Operation Department — Chest X-Ray Classification

> An end-to-end deep learning pipeline for automated chest X-ray diagnosis, with Grad-CAM explainability and Django deployment.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?style=flat-square)
![Django](https://img.shields.io/badge/Django-4.x-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

This project classifies chest X-ray images into four diagnostic categories using a fine-tuned EfficientNetV2S backbone. The full pipeline covers data preprocessing, model training, clinical-grade evaluation, Grad-CAM explainability, and a production-ready Django REST API that doctors can use to upload an X-ray and receive a diagnosis with confidence scores.

---

## Classes

| Label | Description |
|---|---|
| COVID-19 | Bilateral ground-glass opacities |
| Normal | Healthy lung tissue |
| Viral Pneumonia | Diffuse interstitial infiltrates |
| Bacterial Pneumonia | Lobar or segmental consolidation |

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 62.5% (small dataset v2) |
| COVID-19 AUC-ROC | **0.997** |
| Bacterial Pneumonia AUC | 0.830 |
| Macro AUC-ROC | 0.765 |
| Cohen's Kappa | 0.500 |
| COVID-19 F1-Score | 0.95 |

> v3 trained on the full chest_xray dataset (5,216 training images) is expected to reach 88–93% accuracy.

---

## Architecture

```
Input X-Ray (224×224×3)
        │
        ▼
  Center-Crop (5%)           ← removes scanner border artifacts
        │
        ▼
  CLAHE Preprocessing        ← contrast enhancement for lung tissue
        │
        ▼
  EfficientNetV2S Backbone   ← ImageNet pretrained, top 100 layers unfrozen
        │
        ▼
  GeM Pooling                ← Generalized Mean, better than AvgPool for pathology
        │
        ▼
  Dense(512) → BN → Dropout(0.4)
        │
        ▼
  Dense(256) → BN → Dropout(0.2)
        │
        ▼
  Dense(4, softmax)          ← float32 for numerical stability
```

---

## Technical Highlights

### Preprocessing
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) applied on the L channel in LAB color space. `clipLimit=2.0`, `tileGridSize=(8,8)`.
- **Center-crop** of 5% border before CLAHE. Discovered via Grad-CAM analysis that the model was learning scanner border labels instead of lung tissue without this step.

### Training
- **Optimizer:** AdamW with `weight_decay=1e-4`, `clipnorm=1.0`
- **Loss:** Categorical cross-entropy with `label_smoothing=0.1`
- **LR schedule:** Cosine annealing over all epochs
- **Class weights:** Computed automatically with `sklearn.utils.class_weight`
- **Mixed precision:** fp16 via `tf.keras.mixed_precision` — ~2x faster training
- **Early stopping:** Monitored on `val_auc` with `patience=20`
- **Unfrozen layers:** Top 100 layers of EfficientNetV2S backbone

### Evaluation
- Cohen's Kappa (standard clinical agreement metric, target >0.8)
- Per-class AUC-ROC (one-vs-rest)
- Normalised + raw confusion matrices
- ROC curves for all classes
- **Test-Time Augmentation (TTA):** averages predictions over 5 augmented versions per image for +2–5% free accuracy with no retraining

### Explainability
- **Grad-CAM** generates heatmap overlays on the original X-ray, highlighting the regions that drove each prediction. Green border = correct, red border = wrong.

---

## Project Structure

```
AI Operation Department/
├── chest_xray/
│   └── chest_xray/
│       ├── train/          ← 5,216 training images
│       ├── val/            ← 16 images (use validation_split instead)
│       └── test/           ← 624 test images
├── Dataset/                ← original 4-class dataset (532 images)
├── Test/                   ← original test set (40 images)
├── model_export/
│   ├── xray_model.keras
│   └── model_metadata.json
├── AI_Operation_Department_v3.ipynb   ← main notebook
├── best_model_v3.keras
├── training_log_v3.csv
├── training_history_v3.png
├── confusion_matrices_v3.png
├── roc_curves_v3.png
└── gradcam_results_v3.png
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-operation-department.git
cd ai-operation-department

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install tensorflow opencv-python scikit-learn seaborn matplotlib pandas tqdm pillow django
```

---

## Running the Notebook

```bash
jupyter notebook AI_Operation_Department_v3.ipynb
```

Run cells in order. The notebook is split into 14 cells:

| Cell | Purpose |
|---|---|
| 1 | Imports and display setup |
| 2 | Configuration and path verification |
| 3 | CLAHE + center-crop preprocessing |
| 4 | Data generators (train/val split from train folder) |
| 5 | Training batch preview |
| 6 | Build EfficientNetV2S model |
| 7 | Compile and train |
| 8 | Training history plots |
| 9 | Full test set evaluation |
| 10 | Confusion matrices |
| 11 | ROC curves |
| 12 | Test-Time Augmentation (TTA) |
| 13 | Grad-CAM explainability |
| 14 | Final summary |

> **Important:** The `chest_xray/val/` folder only contains 16 images. Cell 4 ignores it and creates a proper 20% validation split from the training data instead (≈1,043 images). Using the 16-image val folder causes early stopping to fire ~40 epochs too early.

---

## Google Colab (Free GPU)

Training on CPU is slow. For best results use Google Colab's free T4 GPU:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload the notebook
3. `Runtime` → `Change runtime type` → `T4 GPU`
4. Upload your dataset to Google Drive
5. Uncomment the Colab cell at the top of the notebook and update `BASE_DIR`:

```python
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/AI Operation Department'
```

---

## Django Deployment

### Export the model from the notebook

```python
import json, os
EXPORT_DIR = os.path.join(BASE_DIR, 'model_export')
os.makedirs(EXPORT_DIR, exist_ok=True)
model.save(os.path.join(EXPORT_DIR, 'xray_model.keras'))
with open(os.path.join(EXPORT_DIR, 'model_metadata.json'), 'w') as f:
    json.dump({'class_names': CONFIG['class_names'],
               'img_size': list(CONFIG['img_size'])}, f)
```

### Django setup

```bash
pip install django
django-admin startproject hospital_ai
cd hospital_ai
python manage.py startapp xray_classifier

# Copy model files
mkdir -p xray_classifier/ml
cp /path/to/model_export/xray_model.keras xray_classifier/ml/
cp /path/to/model_export/model_metadata.json xray_classifier/ml/
```

### Run the server

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/xray/predict/` in your browser.

### API usage

```bash
curl -X POST http://127.0.0.1:8000/xray/predict/ \
  -H "Accept: application/json" \
  -F "xray_image=@/path/to/xray.jpg"
```

Response:

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 94.3,
  "all_probs": {
    "NORMAL": 5.7,
    "PNEUMONIA": 94.3
  },
  "status": "success"
}
```

> **Critical:** The `clahe_preprocess` function in `views.py` must be identical to the one used during training. Different preprocessing = wrong predictions on correct images.

---

## Dataset

This project uses two datasets:

**Original 4-class dataset (Dataset/ folder)**
- 532 images across 4 classes: Covid-19, Normal, Viral Pneumonia, Bacterial Pneumonia
- Used for initial v2 experiments

**Chest X-Ray dataset (chest_xray/ folder)**
- Source: [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 5,216 training images, 624 test images
- 2 classes: NORMAL, PNEUMONIA
- Used for v3 training

**Additional recommended sources for 4-class expansion:**
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — 112,000 images, free
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert) — Stanford, free for research

---

## Known Limitations

- Viral Pneumonia is the hardest class to distinguish — even trained radiologists have difficulty differentiating it from Bacterial Pneumonia visually. The model requires more training data for this class specifically.
- The model has no GPU requirement at inference time, but training on CPU is very slow (~110 seconds per epoch on Apple M-series).
- This is a research/educational project. It is **not** a certified medical device and should not be used as the sole basis for clinical diagnosis.

---

## Version History

| Version | Dataset | Accuracy | Notes |
|---|---|---|---|
| v1 | Dataset/ (532 images) | ~50% | ResNet50, basic augmentation |
| v2 | Dataset/ (532 images) | 62.5% | EfficientNetV2S, CLAHE, GeM pooling, AUC monitoring |
| v3 | chest_xray (5,216 images) | In progress | Center-crop, 100 unfrozen layers, TTA, fixed val split |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Rony** — AI & Software Engineer
[LinkedIn](https://www.linkedin.com/in/rony-zeenaldeen-b288112ab/) · [GitHub](https://github.com/Rony-ZenAlden/)

---

> If you find this useful, please give it a star ⭐ — it helps other developers find the project.
