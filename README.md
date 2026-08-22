# Intelligent Fetal Health Classification

A machine learning project that classifies fetal health status — **Normal**, **Suspect**, or **Pathological** — from cardiotocography (CTG) measurements, using Decision Tree, Random Forest, and XGBoost models.

🔗 **[Try the live demo](#)** *(https://recca-eng-fetal-health-classification--app-rtmz2o.streamlit.app/)*

---

## Project Overview

Fetal health assessment often relies on CTG data, which records fetal heart rate, movement, and uterine activity. This project explores whether machine learning can help classify fetal health status from these measurements, with a particular focus on correctly identifying the clinically critical minority classes — Suspect and Pathological — not just overall accuracy.

## Dataset

The dataset contains 2,126 CTG recordings, each with 21 numerical features and a target label (`fetal_health`: 1 = Normal, 2 = Suspect, 3 = Pathological). After removing 13 duplicate rows, 2,113 observations were used for analysis and modeling.

## Approach

EDA → data cleaning → feature selection (mutual information) → model comparison (Dummy, Decision Tree, Random Forest, XGBoost) → cross-validation & tuning → final model selection → feature importance. Full details and reasoning are in the notebook.

## Results

| Model | Test Accuracy | Test Macro F1 |
|---|---|---|
| Dummy Classifier | 0.78 | 0.29 |
| Decision Tree | 0.93 | 0.88 |
| Random Forest | 0.95 | 0.91 |
| **XGBoost (final model)** | **0.96** | **0.94** |
| XGBoost (tuned) | 0.96 | 0.93 |

## Live Demo

The `app.py` file is a Streamlit app that lets you adjust the model's most influential features and get a live prediction. It uses real dataset statistics (median/min/max) for all inputs  and there are no fabricated values.

**Note:** This tool is for demonstration purposes only. It is not a medical device and should never be used to guide real clinical decisions.



## Limitations

This model has not been clinically validated and should be treated strictly as a research/demonstration project. See [DISCUSSION.md](./DISCUSSION.md) for a full discussion of dataset limitations, generalization concerns, bias, and ethical considerations.

## Tech Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit

## Author

Rebecca Akinboro
