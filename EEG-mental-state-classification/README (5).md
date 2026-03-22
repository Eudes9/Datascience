# VREED — Emotion Recognition from Physiological & Behavioural Signals

Multimodal emotion recognition using eye-tracking, ECG and GSR signals from the [VREED dataset](https://www.kaggle.com/datasets/lumaatabbaa/vr-eyes-emotions-dataset-vreed) (VR Eyes Emotions Dataset).

---

## Context

This project is part of a personal portfolio developed to demonstrate data analysis skills in a research context (target: research engineer position in psychology/neuroscience).

The VREED dataset was collected by Tabbaa et al. (2021) and contains multimodal recordings from 34 participants watching 12 immersive 360° videos in VR, designed to elicit 4 emotional states based on the **Circumplex Model of Affect (CMA)**.

---

## Emotional Classes (Quad_Cat)

| Class | Quadrant | Arousal | Valence | Examples |
|-------|----------|---------|---------|----------|
| 0 | HA+HV | High | Positive | Joy, Excitement |
| 1 | LA+HV | Low | Positive | Calm, Relaxation |
| 2 | LA+LV | Low | Negative | Sadness, Boredom |
| 3 | HA+LV | High | Negative | Fear, Anxiety |

---

## Dataset

| File | Format | Description |
|------|--------|-------------|
| Post-Exposure Ratings | .xlsx | Self-reported labels (valence, arousal, SAM, VAS) |
| Eye Tracking Features | .csv | 49 features (fixations, saccades, blinks, micro-saccades) |
| ECG Features | .csv | 18 features (HRV, RR intervals, LF/HF ratio) |
| GSR Features | .csv | 8 features (peaks, valleys, mean, variance) |

- **312 observations** · **75 features** · **4 balanced classes** (78 obs/class)
- No missing values after median imputation
- Outliers treated by IQR winsorisation

---

## Methodology

```
Raw data
    ↓
Median imputation (missing values)
    ↓
IQR Winsorisation (outliers)
    ↓
Feature selection (RF feature importances)
    ↓
Classification (SVM RBF, Random Forest)
    ↓
Hyperparameter tuning (GridSearchCV, 5-fold CV)
```

---

## Key Finding — 3 Biomarkers Sufficient

Feature importance analysis identified **3 key biomarkers** that alone match the performance of the full 75-feature model:

| Feature | Modality | Description |
|---------|----------|-------------|
| `Number of Peaks` | GSR | Electrodermal response peaks |
| `Number of Valleys` | GSR | Electrodermal response valleys |
| `Num_of_Microsac` | Eye Tracking | Number of micro-saccades |

> GSR peaks/valleys capture **arousal** (physiological activation), while micro-saccades reflect **attentional and emotional state**.

---

## Results

### Test set (80/20 stratified split)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest (75 feat.) | 0.540 | 0.586 | 0.541 | 0.549 |
| SVM RBF (75 feat.) | 0.698 | 0.715 | 0.697 | 0.701 |
| RF Top 3 | 0.587 | 0.602 | 0.593 | 0.572 |
| SVM Top 3 (default) | 0.619 | 0.632 | 0.622 | 0.614 |
| **SVM Top 3 (tuned) ★** | **0.698** | **0.727** | **0.703** | **0.682** |

### Cross-validation (5-fold stratified)

| Model | Accuracy | F1 macro |
|-------|----------|----------|
| Random Forest | 0.622 ± 0.101 | 0.614 ± 0.097 |
| SVM RBF | 0.669 ± 0.092 | 0.661 ± 0.098 |
| RF Top 3 | 0.625 ± 0.071 | 0.618 ± 0.076 |
| **SVM Top 3 (tuned) ★** | **0.698** | **0.682** |

> Baseline random = 0.250 (4 balanced classes)  
> Best model: **~2.8× above chance level**

### Best hyperparameters (GridSearchCV)

```
kernel = rbf · C = 100 · gamma = 0.1
```

### Confusion matrix — SVM Top 3 Tuned

- **Quad1 (LA+HV) : 15/15 — 100%** — calm/relaxation perfectly recognised
- **Quad3 (HA+LV) : 12/16 — 75%** — fear/anxiety well detected
- **Quad0 (HA+HV) : 11/16 — 69%** — joy/excitement correctly classified
- **Quad2 (LA+LV) : 6/16 — 38%** — most difficult class (low activation states physiologically similar)

---

## Project Structure

```
VREED-emotion-recognition/
│
├── data/                          # Raw data (not included)
│   ├── 03 Self-Reported Questionnaires/
│   ├── 04 Eye Tracking Data/
│   └── 05 ECG-GSR Data/
│
├── notebooks/
│   ├── 01_exploration.ipynb       # Data structure exploration
│   ├── 02_import_fusion.ipynb     # Data loading and merging
│   ├── 03_preprocessing.ipynb     # Imputation, winsorisation
│   ├── 04_EDA.ipynb               # Descriptive statistics, correlations
│   └── 05_classification.ipynb    # ML models, tuning, results
│
├── figures/
│   ├── confusion_matrix_svm_top3_tuned.png
│   ├── feature_importances.png
│   └── distribution_y.png
│
└── README.md
```

---

## Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![pandas](https://img.shields.io/badge/pandas-2.0-green)
![seaborn](https://img.shields.io/badge/seaborn-0.12-lightblue)

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

---

## Reference

Tabbaa, L., Searle, R., Ang, C.S., Bafti, S.B., Hossain, M., Intarasirisawat, J., and Glancy, M. (2021). *VREED: Virtual Reality Emotion Recognition Dataset using Eye Tracking & Physiological Measures*. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies.
