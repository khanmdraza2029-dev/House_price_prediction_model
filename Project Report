# 📋 Project Report — House Price Prediction Model

**Repository:** [khanmdraza2029-dev/House_price_prediction_model](https://github.com/khanmdraza2029-dev/House_price_prediction_model)  
**Author:** Mohammad Raza Khan  
**Date:** 2025  

---

## 1. Introduction

Real estate pricing is complex and influenced by dozens of variables — from a property's size and age to its location and proximity to amenities. This project applies supervised machine learning to automate and improve the accuracy of residential property price estimation.

The dataset used is the **Melbourne Housing Market dataset** (`melb_data.csv`), containing 13,580 real property listings across Melbourne, Australia. The target variable is `Price` (in AUD). The complete pipeline — from data loading to model evaluation — is implemented in `Real_Estate_Price_Prediction_model.ipynb`.

---

## 2. Dataset

| Property | Details |
|---|---|
| **File** | `melb_data.csv` |
| **Total Records** | 13,580 |
| **Total Columns** | 21 |
| **Target Variable** | `Price` (AUD) |
| **Mean Price** | $1,075,684 |
| **Min / Max Price** | $85,000 / $9,000,000 |
| **Records after cleaning** | 6,830 (after dropping missing values) |

### Features Used (Auto-detected numeric columns)
`Rooms`, `Distance`, `Postcode`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lattitude`, `Longtitude`, `Propertycount`

### Missing Values Found & Handled

| Feature | Missing Values |
|---|---|
| Car | 62 |
| BuildingArea | 6,450 |
| YearBuilt | 5,375 |

> **Strategy:** Rows with any missing values in selected features were dropped, reducing the dataset from 13,580 to **6,830 clean records**.

---

## 3. Methodology

The project follows a standard supervised ML pipeline:
```
Load Data → EDA → Handle Missing Values → Feature Selection
    → Train/Test Split (80/20) → Model Training → Hyperparameter Tuning
        → Cross-Validation → Evaluation → Insights
```

### Train / Validation Split
| Set | Size |
|---|---|
| Training | 5,464 samples (80%) |
| Validation | 1,366 samples (20%) |
| `random_state` | 1 |

---

## 4. Models & Results

### 4.1 Model 1 — Decision Tree Regressor (Baseline)

A default Decision Tree was trained without any depth constraints as a performance baseline.

| Metric | Value |
|---|---|
| Validation MAE | **$229,232** |

### 4.2 Hyperparameter Tuning — Decision Tree

`max_leaf_nodes` was swept across `[5, 25, 50, 100, 250, 500]` to find the optimal tree size:

| max_leaf_nodes | Validation MAE |
|---|---|
| 5 | $344,615 |
| 25 | $263,011 |
| 50 | $239,002 |
| 100 | $226,176 |
| 250 | $216,263 |
| **500** | **$208,956** ✅ Best |

> **Best max_leaf_nodes = 500**, giving MAE of **$208,956**

### 4.3 Model 2 — Random Forest Regressor (100 trees)

| Metric | Value |
|---|---|
| Validation MAE | **$173,738** |
| Validation RMSE | **$332,026** |
| R² Score | **0.7937** |
| Mean Percentage Error | **16.14%** |

### 4.4 Cross-Validation — Random Forest (n_estimators sweep)

3-fold cross-validation was performed across `n_estimators` values from 50 to 400:

| n_estimators | CV MAE |
|---|---|
| 50 | $213,283 |
| 100 | $212,392 |
| **150** | **$211,795** ✅ Best |
| 200 | $212,433 |
| 250 | $212,529 |
| 300 | $212,316 |
| 350 | $212,383 |
| 400 | $212,035 |

> **Best n_estimators = 150**, giving CV MAE of **$211,795**

---

## 5. Model Comparison Summary

| Model | Validation MAE | Improvement over Baseline |
|---|---|---|
| Decision Tree (Baseline) | $229,232 | — |
| Decision Tree (Optimized, `max_leaf_nodes=500`) | $208,956 | ↓ 8.8% |
| **Random Forest (100 trees)** | **$173,738** | **↓ 24.2%** ✅ |

> ✅ **Best Model: Random Forest** — 24.2% better MAE than the baseline Decision Tree.

---

## 6. Feature Importance (Random Forest)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | BuildingArea | 0.3747 |
| 2 | Distance | 0.1561 |
| 3 | Postcode | 0.1105 |
| 4 | YearBuilt | 0.1011 |
| 5 | Landsize | 0.0681 |
| 6 | Lattitude | 0.0564 |
| 7 | Longtitude | 0.0487 |

> **Key Insight:** `BuildingArea` is the single most important predictor, accounting for ~37% of the model's decision-making, followed by `Distance` from the CBD (~16%).

---

## 7. Kaggle Intermediate ML Exercises

Alongside the main project, the following Kaggle course exercises were completed as part of learning the broader ML workflow:

| Exercise | Topic | Key Skill |
|---|---|---|
| `exercise-introduction.ipynb` | Baseline Model | Random Forest on Ames Housing (Iowa) dataset |
| `exercise-missing-values.ipynb` | Missing Values | Drop, Simple Imputation, Iterative Imputation |
| `exercise-categorical-variables.ipynb` | Categorical Encoding | Drop, Label Encoding, One-Hot Encoding |
| `exercise-pipelines.ipynb` | Pipelines | `sklearn.pipeline.Pipeline` for clean preprocessing |
| `exercise-xgboost.ipynb` | XGBoost | `XGBRegressor`, learning rate, early stopping |
| `exercise-data-leakage.ipynb` | Data Leakage | Target leakage vs. train-test contamination |

All exercises use the **Ames, Iowa Housing Prices dataset** (Kaggle competition: `home-data-for-ml-course`).

---

## 8. Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn (`DecisionTreeRegressor`, `RandomForestRegressor`, `cross_val_score`) |
| Boosting (Kaggle) | XGBoost (`XGBRegressor`) |
| Environment | Jupyter Notebook |

---

## 9. Conclusion

This project successfully demonstrates an end-to-end machine learning pipeline for real estate price prediction. The **Random Forest Regressor** with 100 trees achieved the best performance on the Melbourne dataset, with:

- **MAE of $173,738** on the validation set
- **R² of 0.7937** — explaining ~79% of price variance
- A **24.2% improvement** in MAE over the baseline Decision Tree

The most decisive features were **BuildingArea**, **Distance from CBD**, and **Postcode** — consistent with real-world real estate intuition.

Future improvements could include encoding categorical features (Suburb, Type, Regionname), using XGBoost, or deploying the model as an interactive web application.

---

*GitHub: [@khanmdraza2029-dev](https://github.com/khanmdraza2029-dev)*
