# Project Report — House Price Prediction

---

## The Problem I Chose

I built a machine learning model to predict residential property prices from measurable house attributes. The project covers two datasets — Melbourne property listings (`melb_data.csv`, 13,580 rows, target: `Price` in AUD) for the main notebook, and the Ames, Iowa housing dataset (79 features, ~1,460 samples, target: `SalePrice` in USD) for six Kaggle Intermediate ML course exercises.

---

## Why It Matters

Property prices directly affect buyers, sellers, and lenders, yet traditional appraisal is slow, expensive, and inconsistent. A data-driven model offers consistent, fast, and scalable price estimation. Beyond the application, this problem is ideal for learning ML — it has messy real-world data, mixed feature types, missing values, and a continuous regression target that requires the full pipeline from cleaning to evaluation.

---

## My Approach

**Main project:** Loaded `melb_data.csv`, handled missing values by dropping affected rows (13,580 → 6,830 rows), auto-selected 12 numeric features, split 80/20, then trained and compared a Decision Tree (baseline + tuned) and Random Forest (100 trees + cross-validated). Evaluated on MAE, RMSE, and R².

**Kaggle exercises:** Each notebook introduced one technique — missing value strategies, categorical encoding, sklearn pipelines, XGBoost, and data leakage — and measured its exact effect on MAE, building the full ML skillset step by step.

---

## Key Decisions I Made

**Dropping rows instead of imputing missing values**
`BuildingArea` had 6,450 missing values and turned out to be the most important feature (37% of Random Forest importance). Filling those with the column mean would have injected noise into the model's primary signal. I accepted losing nearly half the dataset in exchange for cleaner, more trustworthy training data.

**Ordinal encoding over one-hot encoding**
After testing all three categorical encoding strategies on the Ames dataset, ordinal encoding produced the lowest MAE:

| Approach | MAE |
|---|---|
| Drop categorical columns | 17,837 |
| One-hot encoding | 17,525 |
| **Ordinal encoding** | **17,098** |

One-hot encoding created too many sparse columns for high-cardinality features, adding noise the model had to work around. This result surprised me and reinforced that empirical testing matters more than theoretical preference.

**Filtering unseen categories before ordinal encoding**
The encoder threw an error when the validation set contained `Condition2` category values not seen during training. I resolved this by only encoding "safe" columns — where all validation categories appeared in training — and dropping the rest. This is a common real-world failure mode that clean tutorial datasets rarely expose.

**n_estimators = 150 for Random Forest**
A 3-fold cross-validation sweep showed gains plateauing after 150 trees, with MAE of $211,795 — lower than both 100 ($212,392) and 200 ($212,433). Beyond 150 trees, training time grew with no meaningful accuracy return.

---

## Challenges I Faced

**Losing half the dataset to missing values**
Dropping rows with any missing numeric feature reduced the dataset by 6,750 rows. Since `BuildingArea` and `YearBuilt` each had thousands of missing entries, the losses compounded. A more sophisticated approach — like predicting `BuildingArea` from other features — could have preserved more data, but was beyond the current scope.

**Imputation performing worse than dropping**
On the Ames dataset, dropping columns with missing values (MAE: 17,837) outperformed mean imputation (MAE: 18,062). This was the opposite of what I expected. The explanation is that mean imputation is a poor fit when the missing data isn't randomly distributed — you can end up adding values that are systematically wrong. It challenged my assumption that retaining more data is always better.

**Understanding data leakage**
The distinction between target leakage (a feature encodes the answer) and train-test contamination (validation data influences training) is subtle and took real effort to internalise. Working through concrete examples — Nike shoelaces, cryptocurrency prices, hospital infection rates — made it click. The danger is that a leaky model can look near-perfect during development while being completely broken in production.

---

## Results

### Melbourne Housing

| Model | MAE | Notes |
|---|---|---|
| Decision Tree (baseline) | $229,232 | No depth constraint |
| Decision Tree (tuned) | $208,956 | `max_leaf_nodes=500` |
| **Random Forest** | **$173,738** | 100 trees, best model |

R²: **0.7937** · RMSE: **$332,026** · Mean error: **16.14%**

Top predictors: `BuildingArea` (0.375) › `Distance` (0.156) › `Postcode` (0.110) › `YearBuilt` (0.101)

### Kaggle — Ames, Iowa

| Approach | MAE |
|---|---|
| Baseline Random Forest | 23,528 |
| Best encoding strategy (ordinal) | 17,098 |
| Pipeline (mean imputation + one-hot + RF) | 17,612 |
| **XGBoost (tuned)** | **17,032** |

---

## What I Learned

**Data preparation dominates.** Cleaning took more time than modelling, and the quality of training data had more impact on results than the choice of algorithm.

**Feature importance reflects reality.** `BuildingArea` and `Distance from CBD` driving Melbourne prices aligns with what any real estate agent would say. When model logic matches domain knowledge, it's a sign the model has learned something genuine.

**Test empirically, not theoretically.** Both times I had a strong theoretical expectation — imputation beats dropping, one-hot beats ordinal — the data disagreed. The data always wins.

**Pipelines are essential.** Applying preprocessing separately to training and validation is error-prone. Bundling everything into a `sklearn.Pipeline` makes the process consistent, cleaner, and safer to deploy.

**XGBoost is meaningfully better.** Sequential boosting (XGBoost, MAE: 17,032) outperformed parallel bagging (Random Forest, MAE: 17,098) consistently on the Ames dataset. Correcting previous errors with each new tree is a more efficient use of model capacity.

**Leakage is the hardest problem to catch.** A model can score perfectly in training while being useless in the real world. Checking whether each feature would actually be available at prediction time should be the first thing you do before building any deployed model.

---
