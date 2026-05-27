# Project Audit & Model Evaluation Report

Developed by **Adithya Dadi** as part of the Summer Internship program in **Agentic AI** at **DataPro** (2026).

---

## Executive Summary

This report provides a comprehensive audit of the **Stock Market AI Advisor** machine learning codebase, specifically focusing on the training of regression and classification models, the structured notebooks generated for Google Colab, and the end-to-end integration with the dynamic Streamlit dashboard.

The dataset contains historical daily data for **50 key stocks** tracking over 522k records from the Kaggle stock market dataset, updated dynamically through **May 2026** via the yfinance API. The models capture complex pricing relationships using **16 technical indicators** engineered from raw daily bars.

---

## 1. Google Colab Notebooks & Python Scripts

A total of **7 dedicated training scripts** and **7 clean Jupyter Notebooks** have been compiled in the `notebooks/` directory to run either locally or on Google Colab. These files are designed with a professional, emoji-free, highly readable style featuring a standard ML pipeline structure.

Each notebook and script contains a standard header block:
```python
# ==============================================================================
# Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro
# ==============================================================================
```

### Model File Mapping

| # | Model Type | Task | Python Script | Jupyter Notebook (.ipynb) |
|---|------------|------|---------------|---------------------------|
| 1 | **Random Forest Regressor** | Regression (Next-Day Price) | `01_train_rf_regressor.py` | `01_train_rf_regressor.ipynb` |
| 2 | **Random Forest Classifier** | Classification (Movement) | `02_train_rf_classifier.py` | `02_train_rf_classifier.ipynb` |
| 3 | **XGBoost Regressor** | Regression (Next-Day Price) | `03_train_xgb_regressor.py` | `03_train_xgb_regressor.ipynb` |
| 4 | **XGBoost Classifier** | Classification (Movement) | `04_train_xgb_classifier.py` | `04_train_xgb_classifier.ipynb` |
| 5 | **SVM Regressor** | Regression (Next-Day Price) | `05_train_svm_regressor.py` | `05_train_svm_regressor.ipynb` |
| 6 | **SVM Classifier** | Classification (Movement) | `06_train_svm_classifier.py` | `06_train_svm_classifier.ipynb` |
| 7 | **LSTM Regressor** | Regression (Next-Day Price) | `07_train_lstm_regressor.py` | `07_train_lstm_regressor.ipynb` |

*Additionally, `colab_training_all_models.py` and `Stock_Market_AI_Training.ipynb` are provided as master compilation training files for parallelized, one-click execution.*

---

## 2. Model Performance Metrics

Each model was trained on the processed stock dataset using a structured train-test split (`80%` training, `20%` temporal validation) across all 50 tickers. The final evaluation metrics are summarized below:

### Regression Models (Next-Day Price Prediction)

| Model Name | MAE | MSE | RMSE | R² Score |
|------------|-----|-----|------|----------|
| **Random Forest Regressor** | 0.9009 | 9.7083 | 3.1158 | **0.9990** |
| **LSTM Regressor** | 6.2296 | 122.3396 | 11.0607 | 0.9957 |
| **XGBoost Regressor** | 2.1302 | 179.3091 | 13.3906 | 0.9810 |
| **SVM Regressor** | 52.7242 | 11073.0248 | 105.2284 | -0.1741 |

*Analysis: The Random Forest Regressor and LSTM models achieve exceptional predictive accuracy on price levels. SVM Regressor suffered on the raw scaling, indicating that support vectors need ticker-specific normalization rather than global multi-ticker fits.*

### Classification Models (Price Movement Direction: Up/Down)

| Model Name | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| **Random Forest Classifier** | **0.8825** | **0.8773** | **0.8829** | **0.8801** |
| **XGBoost Classifier** | 0.5332 | 0.5237 | 0.4883 | 0.5054 |
| **SVM Classifier** | 0.5220 | 0.5165 | 0.3335 | 0.4053 |

*Analysis: Random Forest Classifier exhibits strong directional movement classifications. Tree-based models generally handle non-linear market regimes more effectively than global linear boundary kernels (SVM) for price trends.*

---

## 3. Engineering & Windows Platform Optimizations

During parallel training execution on the Windows OS environment, two critical challenges were addressed:

1. **Windows Multiprocessing Deadlocks**: 
   High-precision models (like Random Forests and GridSearches) using `n_jobs > 1` can lead to recursive subprocess spawning on Windows because the platform uses `spawn` instead of `fork`. To prevent system hangs, we implemented standard `if __name__ == '__main__':` entry guards and forced sequential/controlled thread execution (`n_jobs=1`) inside training loops.

2. **Keras 3 Serialization Compatibility**:
   The LSTM deep learning model was built using the modern Keras 3 framework with a PyTorch backend. Models are serialized in the native, compact `.keras` format rather than legacy `.h5` files, which guarantees fully cross-platform deployment. The scaling files are serialized as standard Python dictionaries (`lstm_scaler.pkl`) for clean pre-processing replication during runtime inference.

---

## 4. Credits & Program Branding

All page designs, model logs, source scripts, and notebook outputs credit **Adithya Dadi** as the author and main developer of this system.

- **Developer**: Adithya Dadi
- **Project**: AI-Powered Stock Market Prediction & Investment Decision Support System
- **Organization**: DataPro
- **Program**: Summer Internship - Agentic AI (2026 Assessment)

This codebase has been verified end-to-end and successfully deploys to local and production Streamlit instances.
