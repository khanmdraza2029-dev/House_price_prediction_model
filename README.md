# 🏠 House Price Prediction Model

A machine learning project that predicts **Melbourne residential property prices** using Decision Tree and Random Forest regression algorithms, trained on the Melbourne Housing dataset.

---

## 📌 About This Project

This project builds a complete supervised ML pipeline in Python to estimate house sale prices based on structural and locational property features. The notebook walks through data loading, missing value handling, feature selection, model training, hyperparameter tuning, cross-validation, and evaluation — all in one place.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **File** | `melb_data.csv` |
| **Source** | Melbourne Housing Market (Kaggle) |
| **Records** | ~13,580 property listings |
| **Target Variable** | `Price` (AUD) |
| **Features Used** | Rooms, Distance, Postcode, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Lattitude, Longtitude, Propertycount |

---

## ⚙️ ML Pipeline

1. **Load Data** — Read `melb_data.csv` via Pandas
2. **Explore** — Inspect shape, columns, data types, and summary statistics
3. **Handle Missing Values** — Drop rows with nulls in selected features
4. **Feature Selection** — Auto-detect all numeric columns as predictors
5. **Train/Validation Split** — 80% train / 20% validation (`random_state=1`)
6. **Model Training** — Decision Tree (baseline + tuned) and Random Forest
7. **Hyperparameter Tuning** — Sweep `max_leaf_nodes` for Decision Tree; sweep `n_estimators` for Random Forest via 3-fold cross-validation
8. **Evaluation** — MAE, RMSE, R² Score
9. **Feature Importance** — Bar chart of top predictors from Random Forest

---

## 🤖 Models Trained

| Model | Description |
|---|---|
| Decision Tree (Baseline) | Default depth, no pruning |
| Decision Tree (Optimized) | Best `max_leaf_nodes` from `[5, 25, 50, 100, 250, 500]` |
| R
