# 🏠 House Price Prediction Model

A machine learning project that predicts residential property prices using the Melbourne Housing Market dataset, built as part of completing Kaggle's Intermediate Machine Learning course.

---

## The Problem

Predicting the price of a residential property is genuinely difficult. A house's value depends on dozens of interacting factors — its size, age, location, number of rooms, distance from the city — and the relationship between these factors and price is rarely linear or obvious. Traditional appraisal relies on human experts, which is slow, expensive, and inconsistent.

This project builds a data-driven solution using supervised machine learning regression. Given a set of measurable property attributes, the model learns the patterns that connect those attributes to sale price, and uses them to predict the price of properties it has never seen before.

The project runs across two parallel tracks:

- **Main project** — Predicting Melbourne property prices using `melb_data.csv` (13,580 real listings, target: `Price` in AUD)
- **Kaggle exercises** — Predicting sale prices of homes in Ames, Iowa using the [Kaggle Housing Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course) dataset (79 features, ~1,460 training samples, target: `SalePrice` in USD)

---

## Why It Matters

Housing is one of the largest financial decisions most people make. A reliable price prediction model gives buyers a benchmark to evaluate whether a listing is fairly priced, helps sellers set competitive prices, and gives lenders a consistent way to assess collateral value — all without waiting for a manual appraisal.

From a learning perspective, this problem is an ideal ML testbed. It has structured tabular data, a mix of numeric and categorical features, significant missing values, and a continuous regression target. Working through it covers nearly every fundamental skill in the ML workflow: cleaning, encoding, imputing, training, tuning, and evaluating.

---

## Approach

### Main Project — Melbourne Housing

The pipeline follows six stages:

**1. Load and Explore**
Loaded `melb_data.csv` into Pandas. The dataset has 13,580 rows and 21 columns. The `Price` column (target) ranges from $85,000 to $9,000,000 with a mean of $1,075,684 — a wide, right-skewed distribution reflecting the mix of affordable and luxury Melbourne properties.

**2. Handle Missing Values**
Four columns had missing data — `Car` (62), `BuildingArea` (6,450), `YearBuilt` (5,375), and `CouncilArea` (1,369). I dropped all rows with any missing values in the selected numeric features, reducing the usable dataset from 13,580 to 6,830 rows. Only numeric columns were used as features, since the notebook's auto-detection step excluded the categorical columns (`Suburb`, `Type`, `Method`, etc.).

**3. Feature Selection**
Auto-detected all numeric columns except `Price` as features: `Rooms`, `Distance`, `Postcode`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lattitude`, `Longtitude`, `Propertycount` — 12 features total.

**4. Train/Validation Split**
80% training (5,464 samples) / 20% validation (1,366 samples), `random_state=1`.

**5. Train and Tune Models**
- Decision Tree — baseline with no depth constraint, then tuned via `max_leaf_nodes` sweep across `[5, 25, 50, 100, 250, 500]`
- Random Forest — 100 trees, then tuned via 3-fold cross-validation sweep of `n_estimators` from 50 to 400

**6. Evaluate**
Compared models on MAE, RMSE, and R² score on the held-out validation set.

---

### Kaggle Exercises — Ames, Iowa Housing

The Kaggle exercises were structured progressively, each one building on the last:

| Exercise | What It Covered |
|---|---|
| Introduction | Compared 5 Random Forest variants, selected best model, submitted to c
