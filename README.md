# Facial Expression Recognition on FER-2013

> A comprehensive deep learning project comparing transfer learning and custom CNN architectures for emotion classification from facial images.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-completed-success.svg)

## 📋 Overview

This project implements a three-stage pipeline for facial expression recognition on the **FER-2013 dataset** (7 emotions: angry, disgust, fear, happy, neutral, sad, surprise). We explore transfer learning with pretrained models and compare against a custom CNN trained from scratch.

**Best Result:** ResNet50 (fine-tuned) achieved **64.6% accuracy** and **0.638 macro-F1** on the official test set.

---

## 🎯 Project Objectives

1. **Exploratory Data Analysis** – Identify class imbalance, data quality issues, and realistic performance targets
2. **Transfer Learning** – Fine-tune three pretrained architectures (VGG16, ResNet50, MobileNetV2)
3. **Custom Architecture** – Train a CNN from scratch and compare performance
4. **Analysis & Comparison** – Quantify the benefits of pretrained vs custom models

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | FER-2013 (Facial Expression Recognition 2013) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| **Total Images** | 35,887 (train: 28,709, test: 7,178) |
| **Format** | 48×48 grayscale JPEG images |
| **Classes** | 7 emotions |
| **Imbalance Ratio** | 16.5:1 (happy vs disgust) |

### Key Findings

- **Severe class imbalance:** Happy class represents 25% of training data; Disgust only 1.5%
- **1,236 duplicate images** in training set
- **39 groups with conflicting labels** (same image, different emotions)
- **568 images leaked from test into training** (7.9% contamination)
- **13 low-contrast/corrupted images**
- **Human ceiling:** ~65% accuracy (Goodfellow et al., 2013)

---

## 🏗️ Project Structure

```
├── Cse366_Submission1.ipynb                           # EDA & Related Work
│   ├── Dataset exploration and visualization
│   ├── Class imbalance analysis
│   └── Data quality checks (duplicates, leakage, conflicts)
│
├── Cse366_Submission2_Pretrained_FineTuning_Group_1.ipynb   # Transfer Learning
│   ├── Fine-tune VGG16, ResNet50, MobileNetV2
│   ├── Overfitting/underfitting diagnosis
│   ├── Performance comparison and ensemble
│   └── Per-class metrics and confusion matrices
│
├── Cse366_Submission3_Custom_CNN_Scratch.ipynb        # Custom Architecture
│   ├── Build 1.77M-parameter CNN from scratch
│   ├── Early stopping and learning rate scheduling
│   ├── Training diagnostics and inference timing
│   └── Comparison with transfer learning results
│
└── PROJECT_SUMMARY.md                                 # Detailed results & insights
```

---

## 📈 Results

### Submission 2: Transfer Learning

Fine-tuned three pretrained models with and without unfreezing the backbone.

| Model | Mode | Test Accuracy | Macro-F1 | Train Time | Leak-Free Acc |
|-------|------|--------------|----------|-----------|---------------|
| VGG16 | Frozen | 0.405 | 0.339 | 11.6 min | 0.402 |
| VGG16 | Fine-tuned | 0.617 | 0.598 | 30.4 min | 0.606 |
| ResNet50 | Frozen | 0.462 | 0.405 | 12.7 min | 0.449 |
| **ResNet50** | **Fine-tuned** | **0.646** | **0.638** | **43.4 min** | **0.630** |
| MobileNetV2 | Frozen | 0.395 | 0.338 | 5.5 min | 0.389 |
| MobileNetV2 | Fine-tuned | 0.608 | 0.578 | 22.0 min | 0.595 |
| **Ensemble (3×FT)** | **Average** | **0.664** | **0.656** | **—** | **0.648** |

**Key Insight:** Fine-tuning unlocked +18–21 percentage points over frozen backbones.

### Submission 3: Custom CNN from Scratch

Trained a 1.77M-parameter custom architecture end-to-end.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 0.586 |
| **Macro-F1** | 0.541 |
| **Leak-Free Accuracy** | 0.572 |
| **Training Time** | 127.0 min |
| **Inference Time** | 1.183 ms/image (845 img/s on CPU) |
| **Epochs Run** | 60 (full convergence) |

**vs. ResNet50 Fine-tuned:** −6.0 pp accuracy gap (pretrained is 10% better).

### Per-Class Performance (Custom CNN)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Angry | 0.584 | 0.578 | 0.581 |
| **Happy** | **0.748** | **0.750** | **0.749** |
| Surprise | 0.576 | 0.557 | 0.566 |
| Neutral | 0.534 | 0.531 | 0.533 |
| Fear | 0.436 | 0.618 | 0.512 |
| Disgust | 0.500 | 0.378 | 0.431 |
| Sad | 0.420 | 0.426 | 0.423 |

**Observations:**
- Happy is easiest (F1: 0.75) – high contrast features
- Fear & Sad hardest to distinguish (F1: 0.51, 0.42) – low-resolution confusion
- Disgust struggles due to severe class imbalance (1.5% of training data)

---

## 🛠️ Architecture & Methodology

### Data Preparation

1. **Cleaning:** Remove 1,275 problematic images (duplicates + conflicts + low-contrast)
2. **Stratified split:** 85% train / 15% validation (preserves class distribution)
3. **Final sizes:** 23,318 training, 4,116 validation, 7,178 test

### Preprocessing

- Pixel scaling: 0–255 → 0–1 (or model-specific normalization)
- Class weights: Inverse frequency (disgust: 10.3×, happy: 0.6×)
- Data augmentation: Horizontal flip, rotation ±15°, shift ±15%, zoom ±15%
- Label smoothing: 0.10–0.15

### Transfer Learning (Submission 2)

**Strategy:** Warm-up classifier head (5 epochs) → unfreeze backbone (25 epochs at lower LR)

- **VGG16:** 14.7M parameters, best single-network result in literature
- **ResNet50:** 23.6M parameters, skip connections enable deep learning
- **MobileNetV2:** 2.3M parameters, lightweight for edge deployment

### Custom CNN (Submission 3)

**Architecture:**

```
Input (48×48×1)
  ↓
Augmentation → Rescaling (1/255)
  ↓
Block 1: Conv(32)×2 → BN → ReLU, MaxPool(2), Dropout(0.25) → 24×24×32
  ↓
Block 2: Conv(64)×2 → BN → ReLU, MaxPool(2), Dropout(0.25) → 12×12×64
  ↓
Block 3: Conv(128)×2 → BN → ReLU, MaxPool(2), Dropout(0.25) → 6×6×128
  ↓
Block 4: Conv(256)×2 → BN → ReLU, MaxPool(2), Dropout(0.25) → 3×3×256
  ↓
Flatten → Dense(256) → BN → ReLU → Dropout(0.5)
  ↓
Dense(7) + Softmax → [Emotion Logits]
```

**Total Parameters:** 1.77 million (no pretrained weights)

**Training Hyperparameters:**
- Optimizer: Adam (LR: 1e-3)
- Batch size: 64
- Loss: Categorical cross-entropy + label smoothing (0.10)
- Early stopping: Patience 10, minimum 5 epochs
- LR schedule: ReduceLROnPlateau (×0.5, patience 4)
- Max epochs: 60

---

## 📚 Key Insights

### 1. Transfer Learning Dominance
Fine-tuned pretrained models outperform custom CNN by ~6 percentage points because ImageNet pre-training provides rich, generalizable feature representations.

### 2. Fine-Tuning is Critical
Frozen backbones vs fine-tuned models show massive gaps:
- VGG16: +21.2 pp
- ResNet50: +18.4 pp
- MobileNetV2: +21.4 pp

### 3. Class Imbalance Impact
Disgust (1.5% of data) achieves only 0.43 F1 despite class weighting. This severely imbalanced class limits overall performance.

### 4. Most Confused Emotion Pairs
| Pair | Confusion Rate | Why |
|------|---|---|
| Fear → Sad | 20.8% | Low-resolution, similar eyebrow/mouth |
| Sad → Neutral | 17.7% | Subtle differences, small image size |
| Disgust → Angry | 31.9% (Custom CNN) | Both involve nose/mouth, extreme imbalance |

### 5. Data Leakage Effect
Official test accuracy (0.646) vs leak-free subset (0.630): **−1.6 pp** inflation from 568 leaked images.

---

## 🚀 Quick Start

### Requirements

```bash
# Core dependencies
tensorflow >= 2.0
keras >= 2.4
numpy >= 1.19
pandas >= 1.1
scikit-learn >= 0.24
matplotlib >= 3.3
seaborn >= 0.11
jupyter >= 1.0
```

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/fer2013-facial-expression-recognition.git
cd fer2013-facial-expression-recognition

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

1. **Submission 1 – EDA & Related Work**
   ```bash
   jupyter notebook Cse366_Submission1.ipynb
   ```
   Explore the dataset, visualize class distributions, identify quality issues.

2. **Submission 2 – Transfer Learning**
   ```bash
   jupyter notebook Cse366_Submission2_Pretrained_FineTuning_Group_1.ipynb
   ```
   Fine-tune VGG16, ResNet50, MobileNetV2. Compare performance and create ensemble.

3. **Submission 3 – Custom CNN**
   ```bash
   jupyter notebook Cse366_Submission3_Custom_CNN_Scratch.ipynb
   ```
   Train custom architecture from scratch. Visualize confusion matrices.

---

## 📖 Recommended Reading Order

1. **Start here:** `PROJECT_SUMMARY.md` – Executive summary, results tables, key insights
2. **Then:** `Cse366_Submission1.ipynb` – Understand the data and challenges
3. **Next:** `Cse366_Submission2_Pretrained_FineTuning_Group_1.ipynb` – See transfer learning in action
4. **Finally:** `Cse366_Submission3_Custom_CNN_Scratch.ipynb` – Compare custom vs pretrained

---

## 💡 Key Takeaways

| Finding | Impact |
|---------|--------|
| ResNet50 beats VGG16 by 2.9 pp | Use ResNet50 for best accuracy |
| Fine-tuning gains +18–21 pp | Always unfreeze backbone after warm-up |
| Pretrained outperforms custom by 6 pp | ImageNet priors are valuable even for faces |
| Ensemble reaches 66.4% (vs 64.6% single) | Combine 3 models for marginal gains |
| Data leakage inflates by ~1.6 pp | Report both official and leak-free metrics |
| Custom CNN: 1.183 ms/image inference | Useful for real-time applications on CPU |

---

## 🔮 Future Improvements

- [ ] **Attention mechanisms** – Focus on critical facial regions (eyes, mouth)
- [ ] **Grad-CAM visualization** – Explain model predictions visually
- [ ] **Larger input size** – Upscale from 48×48 to 96×96 or 224×224
- [ ] **Heavy augmentation** – Increase rotation/shift intensity
- [ ] **Ensemble with diverse architectures** – EfficientNet, Vision Transformer
- [ ] **Cross-dataset validation** – Test generalization on CK+, JAFFE datasets
- [ ] **Knowledge distillation** – Compress ResNet50 into MobileNetV2-sized model
- [ ] **Live webcam demo** – Real-time emotion detection

---

## 📋 Data Leakage Report

**Affected images:** 568 out of 7,178 test images (7.9%)

| Model | Official Test | Leak-Free | Difference |
|-------|---|---|---|
| ResNet50 FT | 0.646 | 0.630 | −1.6 pp |
| Custom CNN | 0.586 | 0.572 | −1.4 pp |

**Recommendation:** Always report leak-free metrics for honest comparison. Official metrics inflate performance by ~1–2 percentage points.

---

## 📚 Related Work

| Paper | Year | Model(s) | Result |
|-------|------|----------|--------|
| Goodfellow et al. | 2013 | CNN contest | Human: 65%, Best: 71% |
| Tang | 2013 | CNN + SVM | 71.2% |
| Pramerdorfer & Kampel | 2016 | VGG/Inception/ResNet ensemble | 75.2% |
| Khaireddin & Chen | 2021 | VGGNet (fine-tuned) | 73.28% |
| Minaee et al. | 2021 | CNN + attention | 70% |

**Our work in context:**
- Single model (ResNet50 FT): **64.6%** – solid baseline
- Ensemble: **66.4%** – approaching published baselines
- Custom CNN: **58.6%** – expected gap from no ImageNet prior

---

## 👥 Author

**Course:** CSE366 (Artificial Intelligence)  
**Date:** November 2026

---

## 📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## 🔗 Resources

- **Dataset:** [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Original Paper:** [Goodfellow et al., ICML 2013](https://arxiv.org/abs/1307.0414)
- **Reference Implementations:**
  - [VGGNet on Keras](https://keras.io/api/applications/vgg/)
  - [ResNet on Keras](https://keras.io/api/applications/resnet/)
  - [MobileNetV2 on Keras](https://keras.io/api/applications/mobilenet/)

---

## 📞 Questions?

For questions about this project, please open an issue on GitHub or contact the team members above.

**Happy emotion recognizing! 😊**
