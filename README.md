# 😴 Lifestyle-Based Early Detection of Sleep Disorder Risk Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow)](https://catboost.ai)
[![Optuna](https://img.shields.io/badge/Optuna-3.0%2B-purple)](https://optuna.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset)

> A comparison of 5 Machine Learning algorithms on a 100,000-record lifestyle dataset to predict sleep disorder risk (**Healthy / Mild / Moderate / Severe**) — no clinical tests required.

---

## 📌 Overview

Sleep disorders are linked to cardiovascular disease, obesity, and mental health issues, yet traditional diagnosis via polysomnography (PSG) is expensive and inaccessible for most. This project builds a multi-class classifier trained on daily lifestyle and physiological data, with **Optuna-based hyperparameter tuning** on the top 2 models.

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | 0.7887 | 0.9295 |
| Random Forest | 0.8715 | 0.9765 |
| AdaBoost | 0.6561 | 0.7271 |
| XGBoost *(tuned)* | 0.9475 | 0.9960 |
| **CatBoost *(tuned)*** | **0.9540** | **0.9974** |

---

## 📊 Dataset

**Source:** [Sleep Health and Daily Performance Dataset – Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset)

- **100,000 records**, 32 features, 0 missing values
- **Target:** `sleep_disorder_risk` → Healthy / Mild / Moderate / Severe
- **Top predictors (CatBoost):** `mental_health_condition`, `sleep_duration_hrs`, `bmi`, `wake_episodes_per_night`, `stress_score`

---

## 📁 Repository Structure

```
├── Code files/
│   └── AML_Mini_Project.ipynb          # End-to-end notebook (EDA → training → tuning → evaluation)
│
├── Dataset/
│   └── sleep_health_dataset.csv        # Raw dataset (100,000 records, 32 features)
│
├── Project Report/
│   └── Sleep_Disorder_Report_Final.docx
│
├── Supporting Diagrams,Visualizations/
│   ├── Confusion_Matrix_-_CatBoost.png
│   ├── Learning_Curve_-_CatBoost.png
│   ├── Feature_Importance_-_CatBoost.png
│   ├── Model_Comparison_Bar_Chart.png
│   └── Updated_model_performances.png
│
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Unreal-coder-1807/Lifestyle-Based-Early-Detection-of-Sleep-Disorder-Risk-using-Machine-Learning---AML.git
cd Lifestyle-Based-Early-Detection-of-Sleep-Disorder-Risk-using-Machine-Learning---AML
pip install -r requirements.txt
jupyter notebook "Code files/AML_Mini_Project.ipynb"
```

---

## 🔬 Methodology

1. **EDA** — Histogram, class distribution, correlation heatmap
2. **Preprocessing** — Label encoding, StandardScaler (LR only), stratified 80/20 split
3. **Training** — 5 models: Logistic Regression, Random Forest, XGBoost, AdaBoost, CatBoost
4. **Tuning** — Optuna (TPE sampler, 5-fold stratified CV) applied to XGBoost & CatBoost
5. **Evaluation** — Weighted Accuracy, Precision, Recall, F1, Specificity, ROC AUC + Confusion Matrix

---

## 📈 Key Results

- **CatBoost** (tuned) — **95.40% accuracy, 0.9974 ROC AUC** — best across all metrics
- No Healthy case ever predicted as Severe (and vice versa) — zero catastrophic misclassifications
- `mental_health_condition` is the single most important feature (CatBoost importance score ~24)

---

## 📄 License

Distributed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for details.

---