# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions on a highly imbalanced dataset.

## 📌 Overview

Fraud represents only ~0.17% of all transactions, which makes it hard to detect. This project covers the full workflow: exploratory data analysis, data cleaning, class balancing with SMOTE, and training and comparing five classifiers using classification reports, ROC curves and confusion matrices.

## 📊 Dataset

- Source: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions × 31 columns, no missing values
- Features: `Time`, `V1`–`V28` (PCA-transformed), `Amount`, and the target `Class` (1 = fraud, 0 = legitimate)
- Only 492 frauds (0.173%) vs 284,315 legitimate transactions
- `Amount` ranges from 0 to 25,691.16 (mean ≈ 88.35)
- `creditcard.csv` is **not included** in this repository (size). Download it from Kaggle and place it in the project root.

## 🔍 Exploratory Data Analysis

`EDA.ipynb` covers dataset shape and types, missing values, descriptive statistics, class imbalance, the `Amount` distribution, transaction times, the correlation heatmap and a Time vs Amount joint plot.

## 🧹 Data Preprocessing

1. Dropped the `Time` column
2. Checked for missing values (none)
3. Standardized `Amount` with `StandardScaler`
4. Removed 9,144 duplicate rows (275,663 transactions left, 473 frauds)
5. Split 80% train / 20% test (`random_state=42`)
6. Balanced the **training set only** with **SMOTE** (220,148 samples per class); the test set keeps its original distribution (55,133 transactions, 91 frauds)

## 🤖 Models & Results

Evaluated on the untouched test set:

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | AUC |
|---|---|---|---|---|
| Decision Tree (max_depth=4) | 0.05 | 0.86 | 0.09 | 0.96 |
| KNN (k=7) | 0.29 | 0.75 | 0.42 | 0.88 |
| SVM (RBF) | 0.14 | 0.73 | 0.23 | 0.93 |
| MLP (64, 32) | 0.66 | 0.75 | 0.70 | 0.94 |
| **Random Forest (100 trees)** | **0.84** | **0.79** | **0.81** | **0.98** |

**Random Forest** gives the best balance between precision and recall and the highest AUC. Because the data is imbalanced, accuracy is not meaningful here: look at the fraud-class precision, recall and F1.

`graphes.ipynb` contains the ROC curves and confusion matrices of all five models.

## 🗂️ Project Structure

```
├── EDA.ipynb            # Exploratory data analysis
├── dataCleaning.ipynb   # Cleaning, scaling, SMOTE, train/test split
├── FinalModelF.ipynb    # Model training, evaluation and saving
├── graphes.ipynb        # ROC curves and confusion matrices
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## ▶️ Usage

1. Download `creditcard.csv` from Kaggle and place it in the project root.
2. Run the notebooks in this order:
   1. `EDA.ipynb`
   2. `dataCleaning.ipynb` → generates `data_train_after_smote.csv` and `data_test.csv`
   3. `FinalModelF.ipynb` → trains the models and saves them as `.joblib` files
   4. `graphes.ipynb` → ROC curves and confusion matrices

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · Matplotlib · Seaborn · joblib

## 🚀 Future Improvements

- Hyperparameter tuning with cross-validation
- Gradient boosting models (XGBoost, LightGBM)
- Decision threshold tuning and precision-recall curves
- Single `Pipeline` (scaler + model) to avoid data leakage

## 👤 Author

**HIBA ELAMINE** – [GitHub](https://github.com/hiba-elamine) 
**KARIMA** 
## 📄 License

This project is licensed under the MIT License.
