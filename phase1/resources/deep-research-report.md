# Executive Summary

This Phase‑1 **“Classical ML” learning pack** equips you with the **core skills and resources** to become a job-ready ML engineer. It covers foundational libraries (NumPy, Pandas, Matplotlib/Seaborn, Scikit-Learn), key concepts (data loading/cleaning, preprocessing, modeling, evaluation), and two end-to-end projects (House Price Regression and Titanic/Churn Classification). You will also learn to deploy models via Docker and FastAPI, follow a **3-week Beastburner schedule** with daily milestones, and prepare a strong GitHub portfolio with projects, README, and badges. The content below provides **curated study materials**, **code templates**, **project blueprints**, **deployment instructions**, a **day-by-day plan**, and **practice questions** – all with links to official docs and tutorials for further depth.

**Key takeaways:** Focus on hands-on practice (build real projects and GitHub commits daily). Start with data handling and simple models (Linear/Logistic Regression, Decision Trees), master evaluation metrics (accuracy, confusion matrix, RMSE), then advance to Random Forest/XGBoost. Use pipelines for clean code. Finally, deploy one model with FastAPI/Docker to demonstrate production skills. Continuous iteration (build-evaluate-improve) and documentation (README, well-commented code) are crucial for employability. 

# Prioritized Study Materials

Below are **top official and tutorial resources** (mostly documentation and concise guides) for each core topic. These should be your first go-to references before deep-diving into projects.

| **Topic** | **Resource** | **Type** | **Notes** |
|---|---|---|---|
| **NumPy** | Official “Absolute Beginners” (NumPy User Guide)【1†L72-L80】 | Official docs | Explains `ndarray` and basic operations. |
| &nbsp; | “NumPy Quickstart Tutorial” (NumPy.org) | Official tutorial | Quick examples of array creation, indexing, math operations. |
| &nbsp; | *Optional:* DataCamp “Intro to NumPy” (free) | Online course | Hands-on coding practice (requires sign-up). |
| **Pandas** | “10 Minutes to pandas” (Pandas docs)【3†L73-L80】 | Official guide | Concise intro to `Series`, `DataFrame`, indexing, basic ops. |
| &nbsp; | Pandas Cookbook (User Guide) | Official docs | Recipes for data cleaning, grouping, merging, missing data. |
| &nbsp; | *Optional:* Kaggle “Pandas Tutorial” (free) | Tutorial notebook | Practical examples with Titanic or Uber data. |
| **Matplotlib / Seaborn** | Matplotlib Quickstart【32†L130-L134】 | Official tutorial | Basic plotting with `plt.subplots()`, `ax.plot()`. |
| &nbsp; | Seaborn Official Tutorial | Official docs | High-level plots (e.g., `sns.histplot`, `sns.boxplot`). |
| &nbsp; | “Data Visualization with Matplotlib/Seaborn” (YouTube/Blog) | Tutorial | Walkthrough of common chart types (histograms, scatter, bar). |
| **Scikit-Learn (ML)** | “Intro to ML with scikit-learn” (Scikit-Learn docs)【5†L65-L74】【5†L78-L81】 | Official tutorial | Defines supervised vs unsupervised, classification vs regression. |
| &nbsp; | Scikit-Learn User Guide (Supervised Learning) | Official docs | Detailed guides on regressors/classifiers and usage patterns. |
| &nbsp; | *Optional:* Kaggle “Intro to Machine Learning” | Tutorial notebook | Practical use of scikit-learn on toy dataset. |
| **Evaluation Metrics** | Scikit-Learn Metrics docs (e.g., `accuracy_score`, `confusion_matrix`)【24†L676-L684】【26†L679-L688】 | Official docs | Definitions and examples of accuracy, confusion matrix, etc. |
| &nbsp; | Classification Metrics (blog)【22†L56-L64】【22†L95-L104】 | Tutorial article | Illustrates accuracy, confusion matrix, precision/recall with examples. |
| &nbsp; | Regression Metrics (scikit docs) | Official docs | MSE, RMSE, MAE (see User Guide). |

> **Sources:** Official docs and guides (NumPy【1†L72-L80】, Pandas【3†L82-L90】, Scikit-Learn【5†L65-L74】) provide authoritative introductions to each tool. Additional tutorials and notebooks (Kaggle, Real Python, etc.) can be consulted for practice after mastering the basics.

# Step-by-Step Code Templates

Below are **copy-ready Python code snippets** for common tasks in Phase 1. You can adapt them in Jupyter notebooks or scripts.

```python
# Basics: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

## Data Loading & Cleaning

```python
# Load CSV into Pandas DataFrame
df = pd.read_csv('data/house_prices.csv')   # use your path
print(df.head())        # show first rows
print(df.info())        # columns and types
print(df.describe())    # summary stats

# Check missing values
print(df.isnull().sum())
# Drop or fill missing if any (example: drop rows with missing)
df = df.dropna()        # simple drop (or use fillna)
```

> **Comments:** Always inspect `df.info()` and `df.isnull().sum()` to identify missing values and data types. Use `df.dropna()` or `df.fillna()` as needed.

## Exploratory Data Analysis (EDA)

```python
# Univariate analysis
df.hist(figsize=(8,6)); plt.tight_layout()  # histograms of numeric columns
plt.show()

# Correlation heatmap (for numeric features)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Pairplot of a few features (example)
sns.pairplot(df[['Price','SqFt','Bedrooms','Bathrooms']])
plt.show()
```

> **Comments:** Use histograms to view distributions (skewness, outliers). Heatmaps show correlations (great for regression projects). `sns.pairplot` helps visualize pairwise relationships.

## Preprocessing (Scaling, Encoding, Pipelines)

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Example: Separate features/target
X = df.drop('Price', axis=1)   # drop target for regression
y = df['Price']

# Identify numeric vs categorical columns
num_cols = X.select_dtypes(include=['int64','float']).columns
cat_cols = X.select_dtypes(include=['object','category']).columns

# Pipeline: Scale numeric, encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# Example pipeline with a placeholder model (to be inserted later)
from sklearn.linear_model import LinearRegression
pipeline = Pipeline([
    ('prep', preprocessor),
    ('reg', LinearRegression())
])

# Apply preprocessing to training data only later via pipeline.fit()
```

> **Comments:** Use `StandardScaler` to normalize numeric features; use `OneHotEncoder` (or label encoding) for categorical. `ColumnTransformer` lets you apply these selectively. Wrapping everything in a `Pipeline` (sklearn.pipeline.Pipeline) ensures clean chaining of steps【30†L679-L688】【30†L772-L784】.

## Training Models

### Linear Regression (Regression)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
```

> **Comments:** `LinearRegression` assumes numeric target. Always split into train/test. Evaluate with RMSE/MAE for regression (lower is better).  

### Logistic Regression (Classification)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Assume X,y are from a classification dataset (target binary 0/1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

> **Comments:** For binary classification, `accuracy_score` and `confusion_matrix` are basic evaluation tools. (See Scikit docs【26†L679-L688】.)  

### Decision Tree & Random Forest

```python
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Example: Regression tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_tree = dt.predict(X_test)

# Random Forest (regressor)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

> **Comments:** Decision Trees handle non-linear patterns. Random Forest (ensemble of many trees) usually improves performance by averaging. For classification tasks, use `DecisionTreeClassifier` and `RandomForestClassifier`.  

### XGBoost (Gradient Boosting)

```python
# If XGBoost is installed
import xgboost as xgb
xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)
```

> **Comments:** XGBoost often yields high accuracy in both regression and classification【4†L13-L17】. Install via `pip install xgboost`. (Alternatively use `GradientBoostingRegressor` from sklearn.)

## Evaluation (Metrics)

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

# Classification metrics example
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Regression metrics (already showed RMSE/MAE above)
```

> **Comments:** In imbalanced classification, rely on precision/recall/F1 (scikit functions) besides accuracy. ROC AUC measures overall ranking performance. Use `confusion_matrix` to see TP/TN/FP/FN counts【26†L679-L688】.  

## Model Persistence (Saving & Loading)

```python
import joblib
# Save model to file
joblib.dump(model, 'models/linear_model.pkl')
# Later, load it
loaded_model = joblib.load('models/linear_model.pkl')
```

> **Comments:** Use `joblib` or `pickle` to save your trained models for later use (e.g., in a web service). This preserves the exact trained parameters.

# Project Blueprints

Two full projects are outlined below. Each is an end-to-end ML task with a provided dataset link, code structure, and deliverables. Ensure you **document and commit frequently** as you build.

## Project 1: House Price Regression

- **Dataset:** Example CSV with house features and price (e.g., the sample [house-prices.csv] raw file【13†L1-L9】). Columns: `Home, Price, SqFt, Bedrooms, Bathrooms, Offers, Brick, Neighborhood`. (Alternatively use [Kaggle’s House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) or [King County Houses](https://www.kaggle.com/harlfoxem/housesalesprediction).)
- **Goal:** Predict `Price` from other features.

**Repository Structure (Phase-1-ML/Project-1-House-Price):**

```
Project-1-House-Price/
│
├── data/
│   └── house-prices.csv         # (raw data file, see link)
├── notebooks/
│   ├── 1_data_exploration.ipynb  # EDA: summary, visualization
│   └── 2_model_build.ipynb       # Modeling: train/test, fit models, evaluate
├── models/
│   └── price_model.pkl           # (saved final model)
└── README.md                     # Project description, results
```

**Key Steps:**

1. **Load & Explore Data:** (Notebook 1) Inspect `df.describe()`, distributions, correlations. For example, plot Price vs SqFt scatter, histogram of prices.
2. **Preprocess:** Handle missing (if any), encode `Neighborhood` (one-hot), scale numeric.
3. **Baseline Model:** Fit **Linear Regression**. Evaluate with RMSE/MAE. Report error.
4. **Improved Models:** Try **Random Forest** and **XGBoost**. Compare performance. Use cross-validation or train/test split.
5. **Results:** Save best model (`price_model.pkl`). Plot actual vs predicted price scatter.
6. **Documentation:** In `README.md`, describe data source, features, selected model, final error metrics. Include a sample prediction.

**Expected Output:** Clean notebook(s) with plots and printed metrics. Final RMSE/MSE value. README summary.

**Git Commit Checklist:**
- `[ ]` Initial commit: empty repo + `.gitignore`.
- `[ ]` Add raw data (or link) + data exploration notebook.
- `[ ]` Add preprocessing and baseline model code.
- `[ ]` Add advanced model and evaluation improvements.
- `[ ]` Final commit: include model save, README, results.

## Project 2: Titanic (Classification) *or* Customer Churn

### Option A: Titanic Survival Prediction

- **Dataset:** [Titanic train.csv] (raw from GitHub【19†L1-L4】 or Kaggle [link](https://www.kaggle.com/c/titanic)).
- **Goal:** Predict `Survived` (0/1) from passenger features.

**Repo Structure (Phase-1-ML/Project-2-Titanic):**

```
Project-2-Titanic/
│
├── data/
│   └── train.csv            # Titanic data (provided)
├── notebooks/
│   ├── 1_data_preprocessing.ipynb  # handle age/fare missing, encode gender/embarked
│   └── 2_modeling.ipynb            # fit LogisticRegression, RandomForest, etc.
├── models/
│   └── titanic_model.pkl      # (saved model)
└── README.md                   # Summary of approach and accuracy
```

**Key Steps:**

1. **Data Prep:** (Notebook 1) Handle missing `Age` (impute median), drop or encode `Cabin`, fill `Embarked` mode, encode `Sex` and `Embarked`.
2. **Baseline Model:** Train **Logistic Regression**. Evaluate **accuracy**, show confusion matrix【26†L679-L688】.
3. **Improvement:** Try **Random Forest**. Check if accuracy improves. Possibly tune parameters briefly.
4. **ROC Curve:** (optional) Compute ROC AUC for logistic (using `predict_proba`).
5. **Results:** Save final model (`titanic_model.pkl`). Summarize accuracy (~0.8-0.83 typical).
6. **README:** Summarize feature engineering, best accuracy, and any caveats.

**Expected Output:** Accuracy (~80%+), confusion matrix, or classification report. README summary.

**Git Commits:**
- `[ ]` Add Titanic data & initial EDA.
- `[ ]` Add preprocessing code and baseline model.
- `[ ]` Add improved model and final metrics.
- `[ ]` Final commit: model file, README.

### Option B: Customer Churn Prediction

- **Dataset:** Sample [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (or any churn dataset).
- **Goal:** Classify customers as churn/no-churn.

**Structure and steps similar to Titanic**: data cleaning, encode categorical (lots of string fields), train/test split, logistic/regression or tree-based models, evaluate accuracy and ROC.

**Dataset Links:** (Use Kaggle if possible, or search for “Telco Churn dataset”).

# Deployment: Docker + FastAPI Template

After training a model, you can **serve it via a simple API**. Below is a template FastAPI app and Docker instructions.

```python
# app.py (FastAPI web service)
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('models/price_model.pkl')  # load your trained model

@app.get("/")
async def root():
    return {"message": "Model API is up."}

@app.post("/predict")
async def predict(features: dict):
    """
    Expect JSON body with feature values, e.g. {"SqFt": 2000, "Bedrooms":3, ...}.
    """
    # Example: assemble features in order expected by model
    x = np.array([[
        features["SqFt"], features["Bedrooms"], features["Bathrooms"],
        features["Offers"], 1 if features["Brick"] == "Yes" else 0,
        # plus one-hot for Neighborhood if needed...
    ]])
    y_pred = model.predict(x)
    return {"prediction": y_pred[0].item()}
```

**Dockerfile example:**

```
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /app/
COPY models/ /app/models/
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Commands to build/run:**

```bash
docker build -t ml-api .
docker run -d -p 8000:8000 ml-api
```

This container will serve your model on port 8000. In FastAPI, `/predict` expects JSON features and returns predictions. (If deploying to the cloud, adapt to AWS/Azure/Heroku as needed.)

> **Note:** Adjust feature ordering in `app.py` to match your model’s expectations. Use `joblib.dump` paths consistent with Docker build.

# 3-Week Beastburner Schedule (Day-by-Day)

Follow this **intensive 3-week plan** with daily goals. Commit early and often (commit messages in quotes below):

```mermaid
gantt
    title 3-Week Phase-1 ML Plan
    dateFormat  YYYY-MM-DD
    section Week 1: Foundations (Days 1-7)
    Day 1-2: Python & NumPy basics          :done,    d1, 2026-04-15, 2d
    Day 2-3: Pandas basics & small EDA      :done,    d3, 2026-04-17, 2d
    Day 4-5: Train/Test split & first model (Linear Regression): d5, 2026-04-19, 2d
    Day 6-7: Evaluate model (RMSE), report : d7, 2026-04-21, 2d
    section Week 2: Advanced ML (Days 8-14)
    Day 8-9: Learn RandomForest / XGBoost     : d9, 2026-04-23, 2d
    Day 10-11: Train tree-based models, compare : d11, 2026-04-25, 2d
    Day 12-13: Metrics (MAE, RMSE, plots)     : d13, 2026-04-27, 2d
    Day 14: Finalize Project 1 (README, commit): d14, 2026-04-28, 1d
    section Week 3: Second Project & Prep (Days 15-21)
    Day 15-16: Titanic data cleaning & EDA     : d16, 2026-04-30, 2d
    Day 17-18: Logistic Regression for Titanic: d18, 2026-05-02, 2d
    Day 19: RandomForest/Cross-val on Titanic   : d19, 2026-05-03, 1d
    Day 20: Deployment (FastAPI + Docker)     : d20, 2026-05-04, 1d
    Day 21: Portfolio polishing & README      : d21, 2026-05-05, 1d
```

- **Week 1:** Focus on data handling and a simple regression model. *Commits:* “initial data load + EDA”, “first linear regression fit”, “evaluation metrics” .  
- **Week 2:** Add complex models and thorough evaluation. *Commits:* “random forest model added”, “compare models”, “Project1 complete”.  
- **Week 3:** Start second classification project. Add deployment. *Commits:* “titanic data prep”, “titanic logistic model”, “API deployment setup”, “final README & polish”.  

# Practice Exercises & Interview Prep

Use these **short tasks and Q&A** to test your understanding:

- **Mini Exercises:**  
  - *Load & Describe:* Load a CSV of any small dataset. Print `df.head()`, `df.describe()`, and missing value counts.  
  - *Visualization:* Plot a histogram of one feature and a scatterplot of two features using Matplotlib/Seaborn.  
  - *Scaling:* Manually apply `StandardScaler` to a numeric array and show mean=0, std=1.  
  - *Encoding:* Convert a small categorical list to numeric using one-hot or label encoding.  
  - *Modeling:* On Iris dataset (available in sklearn), train a Logistic Regression and compute accuracy & confusion matrix.  

- **Interview Questions:**  
  - *Explain*: What is the difference between supervised and unsupervised learning【5†L65-L74】? (Answer: supervised has input-output pairs for training; unsupervised has no labels.)  
  - *Define*: Overfitting vs underfitting (solution: think bias-variance).  
  - *Why scale data?* (Answer: helps gradient-based models converge, ensures features contribute equally).  
  - *What’s a confusion matrix?* (We compute it to analyze TP/FP/FN/TN【26†L679-L688】.)  
  - *Describe* the steps in cross-validation (e.g., k-fold split, train/test, average metrics).  
  - *SQL / DSA:* (Basic SQL joins, Python list/dict operations—most ML interviews expect these too.)  
  - *Project Discussion:* Be prepared to explain any of your project steps and results (feature choices, model pros/cons, etc.).  

Use online quizzes (e.g. Kaggle Learn quizzes) or prepare brief answers to these to solidify understanding.

# Resources Comparison Tables

**Learning Resources Comparison** (top picks):

| Library       | Resource                                | Format        | Highlights                                           |
|---------------|-----------------------------------------|---------------|------------------------------------------------------|
| NumPy         | NumPy User Guide (beginner)【1†L72-L80】 | Official Docs | Introduces `ndarray`, array ops (official overview). |
|               | NumPy Quickstart (numpy.org)            | Official Docs | Quick examples: array creation, slicing, vector math.|
| Pandas        | 10 min to pandas【3†L73-L81】           | Official Docs | Core `Series`/`DataFrame`, essential operations.     |
|               | Pandas user guide                       | Official Docs | Deeper recipes: grouping, merging, handling missing. |
| Scikit-Learn  | Getting Started tutorial【5†L65-L74】    | Official Docs | ML types (regression/classification) and datasets.    |
|               | User Guide (Supervised Learning)        | Official Docs | In-depth on each algorithm.                          |
| Matplotlib    | Quickstart tutorial【32†L130-L134】      | Official Docs | Plot creation example (`subplots()`, `plot()`).      |
| Seaborn       | Seaborn tutorials                       | Official Docs | High-level statistical plots (hist, box, pairplot).  |

**Project Datasets Comparison:**

| Project       | Dataset                         | Source (Link)                        | Size     | Key Features      |
|---------------|---------------------------------|--------------------------------------|----------|-------------------|
| House Price   | house-prices.csv               | (Example raw CSV)【13†L1-L9】 or Kaggle | ~100 rows | SqFt, Beds, Baths, Offers, Brick, Neighborhood |
| Titanic       | train.csv (Titanic ML)         | [Kaggle Titanic](https://kaggle.com) or GitHub【19†L1-L4】 | 891 rows  | Pclass, Sex, Age, SibSp, Parch, Fare, Embarked |
| Churn         | Telco Customer Churn          | [Kaggle Churn](https://kaggle.com)     | ~7K rows  | Tenure, Services, Charges, Contract type |

*Choose a dataset that’s manageable in size and well-structured. The ones above have plenty of features for ML practice.*

**Model Choice Summary (for classification vs regression):**

| Task       | Models (start with)           | Notes                                          |
|------------|-------------------------------|------------------------------------------------|
| Regression | Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost | Compare RMSE/MAE. RF/XGB often improve over linear on complex data. |
| Classification | Logistic Regression, Decision Tree Classifier, Random Forest, XGBoost, SVM | Check accuracy and ROC. Use RF/XGB for higher complexity. |

# README & Badges (Example)

Your **README.md** should showcase professionalism. Include:

- **Project Title & Objective:** “House Price Prediction – Regression Project”  
- **Tech Stack:** Python, NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn  
- **Project Overview:** “Prediction of house sale prices using regression models.”  
- **Usage:** How to run notebooks or scripts.  
- **Results:** Highlight final model accuracy (e.g., RMSE) or classification accuracy.  
- **Structure:** Briefly list repo folders (data, notebooks, models).  
- **Badges:** e.g., Build Passing, Python Version, License. Use [shields.io](https://shields.io) links.  
- **Contact / Links:** Your GitHub (and LinkedIn profile link).

**Example README snippet:**  

```markdown
# House Price Prediction

**Phase-1 ML Project:** Predict house prices using Linear Regression and Random Forest.  

- **Data:** 100 house records with size, bedrooms, bathrooms, etc.【13†L1-L9】.  
- **Models:** Trained LinearRegression and RandomForestRegressor. Best RMSE achieved: 7,243.  
- **Usage:** See `notebooks/` for code. Run `pip install -r requirements.txt` to install dependencies.  

![Python](https://img.shields.io/badge/python-3.9-blue) ![License](https://img.shields.io/badge/license-MIT-green)
```

Include badges in Markdown via URLs (for example, Python version and license badges). This makes your repo look polished.

# References

All code patterns and definitions above are based on official documentation and tutorials. In particular:

- NumPy Basics (official guide)【1†L72-L80】  
- Pandas Quickstart (10min)【3†L82-L90】  
- Scikit-Learn Intro and Pipeline docs【5†L65-L74】【30†L679-L688】  
- Evaluation metrics explanation【22†L56-L64】【26†L679-L688】  

Use these references for deeper reading. Good luck on your AI/ML journey – focus on **consistent building and documenting** and you’ll be well ahead!