# Project Report — House Price Prediction

**Author:** Mohammad Raza Khan
**GitHub:** [khanmdraza2029-dev](https://github.com/khanmdraza2029-dev)
**Repository:** [House_price_prediction_model](https://github.com/khanmdraza2029-dev/House_price_prediction_model)

---

## 1. The Problem I Chose

I chose to build a machine learning model that predicts residential property prices based on physical and locational features of a house. Specifically, the project works across two datasets and two scopes:

- **Main project (`Real_Estate_Price_Prediction_model.ipynb`):** Predicting Melbourne property prices using the `melb_data.csv` dataset, covering 13,580 real listings across Melbourne, Australia. The target variable is `Price` in Australian Dollars.
- **Kaggle exercises:** Predicting sale prices of homes in Ames, Iowa using the Kaggle *Housing Prices Competition for Kaggle Learn Users* dataset — a structured dataset with 79 explanatory variables and ~1,460 training samples.

---

## 2. Why It Matters

Housing is one of the most significant financial decisions a person makes in their lifetime, yet price estimation is traditionally done by human appraisers — a process that is slow, expensive, and prone to inconsistency. An accurate ML-based prediction model has real value for buyers trying to assess fair market value, sellers setting listing prices, and banks conducting loan risk assessments.

Beyond the real-world application, this problem is well-suited for learning the core machine learning workflow. It involves structured tabular data, a mix of numeric and categorical features, missing values, and a continuous prediction target — making it an ideal environment to understand and practice every stage of an ML pipeline, from raw data all the way to model evaluation.

---

## 3. My Approach to Solving It

I approached the problem in two parallel tracks — a hands-on main project notebook and a structured learning path through Kaggle's Intermediate Machine Learning course exercises.

### Track 1 — Main Project (Melbourne Dataset)

The main notebook follows a complete end-to-end pipeline:

**Step 1 — Data Loading & Exploration**
Loaded `melb_data.csv` using Pandas. The dataset has 13,580 rows and 21 columns. The target column `Price` has a mean of $1,075,684, a minimum of $85,000, and a maximum of $9,000,000, indicating a wide and right-skewed distribution of property values.

**Step 2 — Feature Selection**
I used auto-detection to select all numeric columns as features, excluding the target. This gave 12 features: `Rooms`, `Distance`, `Postcode`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lattitude`, `Longtitude`, and `Propertycount`.

**Step 3 — Handling Missing Values**
Three features had missing data: `Car` (62 missing), `BuildingArea` (6,450 missing), and `YearBuilt` (5,375 missing). I dropped all rows with any missing values in selected features. This reduced the usable dataset from 13,580 rows to **6,830 rows**.

**Step 4 — Train/Validation Split**
Split the data 80/20 — 5,464 training samples and 1,366 validation samples — using `random_state=1` for reproducibility.

**Step 5 — Model Training & Tuning**
Trained a Decision Tree as a baseline, then tuned it, then trained a Random Forest. I also ran 3-fold cross-validation to find the optimal number of trees for the Random Forest.

**Step 6 — Evaluation**
Evaluated all models using MAE, RMSE, and R² on the held-out validation set.

---

### Track 2 — Kaggle Intermediate ML Exercises (Ames, Iowa Dataset)

These exercises built progressively on each other, each introducing a new technique on the same housing dataset:

| Exercise | Technique Learned |
|---|---|
| Introduction | Comparing 5 Random Forest variants, selecting best model |
| Missing Values | Drop columns vs. Mean Imputation — compared via MAE |
| Categorical Variables | Drop vs. Ordinal Encoding vs. One-Hot Encoding — compared via MAE |
| Pipelines | Bundling preprocessing + model into a `sklearn.Pipeline` |
| XGBoost | Training gradient boosted trees, tuning `n_estimators` and `learning_rate` |
| Data Leakage | Identifying and preventing target leakage and train-test contamination |

---

## 4. Key Decisions I Made

### Decision 1 — Drop rows with missing values (Main project)

When I found that `BuildingArea` and `YearBuilt` had over 5,000 missing values each, I had a choice: impute or drop. I chose to drop rows with missing values because these two features turned out to be highly important to the model (BuildingArea alone accounts for 37% of the Random Forest's feature importance). Imputing a physically meaningful quantity like building area with the column mean would have introduced significant noise and distorted what the model learns.

The trade
