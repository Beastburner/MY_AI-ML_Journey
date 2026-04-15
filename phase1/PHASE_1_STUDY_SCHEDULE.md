# Phase 1: Classical Machine Learning - Specific Study Schedule
## Based on your deep-research-report.md Timeline

> **From your report**: "Estimated Timeline: ~4–6 weeks total, including 1–2 small projects (one regression, one classification). For example, spend 1–2 weeks on scikit-learn basics and linear models, 1–2 weeks on tree ensembles and evaluation techniques, and 1–2 weeks building/iterating on a project."

## 📅 Weekly Breakdown (4-6 Week Plan)
**Recommended**: 15-20 hours/week (2-3 hours/day, 5-6 days/week)

### **Week 1-2: Scikit-learn Basics & Linear Models**
**Goal**: Master scikit-learn workflow and linear algorithms

**Days 1-2: Environment & Workflow**
- Setup: Anaconda environment, JupyterLab, GitHub repo
- Learn: scikit-learn API consistency (fit/predict/score)
- Practice: Loading datasets (iris, diabetes, breast cancer)
- Mini-task: Create train/test split, train LinearRegression, evaluate

**Days 3-4: Linear Regression Deep Dive**
- Theory: Cost function, gradient descent, normal equation
- Practice: LinearRegression, Ridge, Lasso on multiple datasets
- Focus: Regularization effects, feature scaling importance
- Visualization: Coefficient plots, learning curves

**Days 5-6: Logistic Regression & Classification Basics**
- Theory: Sigmoid function, log loss, decision boundaries
- Practice: LogisticRegression on binary/multi-class problems
- Learn: Probability outputs vs class predictions
- Metrics: Accuracy, precision, recall, F1, ROC-AUC introduction

**Day 7: Review & Mini-quiz**
- Create one-page cheat sheet for linear models
- Quiz: When to use Ridge vs Lasso, interpreting coefficients
- Plan: Which regression project to start (House Prices recommended)

### **Week 3-4: Tree Ensembles & Evaluation Techniques**
**Goal**: Master non-linear models and rigorous evaluation

**Days 1-2: Decision Trees Fundamentals**
- Theory: Entropy, Gini impurity, information gain, tree depth
- Practice: DecisionTreeRegressor/Classifier, visualize trees
- Learn: Overfitting with deep trees, underfitting with shallow
- Technique: Cost complexity pruning (ccp_alpha)

**Days 3-4: Random Forests & Ensemble Methods**
- Theory: Bootstrap aggregating (bagging), feature randomness
- Practice: RandomForestRegressor/Classifier, feature importance
- Compare: Single tree vs forest performance
- Hyperparameter: n_estimators, max_depth, max_features

**Days 5-6: Advanced Models & Evaluation Mastery**
- Practice: KNN (curse of dimensionality), SVM (kernel trick)
- Introduction: XGBoost/LightGBM (gradient boosting concepts)
- Deep dive: Cross-validation (K-fold, stratified), learning curves
- Metrics deep dive: MAE/RMSE for regression, precision-recall tradeoff

**Day 7: Evaluation Project Planning**
- Finalize project choice and scope
- Set up GitHub repository for project tracking
- Plan preprocessing steps needed

### **Week 5-6: Capstone Project(s)**
**Goal**: Build end-to-end ML pipeline(s) with documentation

**Option A: Two Sequential Projects (Recommended)**
- **Project 1 (Regression)**: House Price Prediction (Weeks 5)
- **Project 2 (Classification)**: Titanic Survival OR Customer Churn (Week 6)

**Option B: One Integrated Project (Alternative)**
- Build both regression and classification components in one project
- Example: Predict house price AND classify price category

**Week 5 Activities**:
- Data acquisition and initial EDA
- Baseline model establishment
- Preprocessing pipeline development
- First model iterations

**Week 6 Activities**:
- Feature engineering experiments
- Model comparison and selection
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Final evaluation and documentation
- GitHub repository completion with README

## 🎯 Daily Time Allocation Template
```
Hours 1-1.5: Focused learning (video/tutorial reading)
Hours 1.5-2.5: Hands-on coding practice
Hours 2.5-3: Review, notes, and planning next steps
```
*Adjust based on your energy levels - mornings often best for new concepts*

## 📊 Progress Tracking Metrics
Track these in your GitHub repo README or a simple log:

| Week | Target Competency | Evidence of Mastery |
|------|-------------------|---------------------|
| 1    | scikit-learn workflow | Can load data, split, train LinearRegression, predict, score without help |
| 2    | Linear models mastery | Understands regularization, can choose Ridge/Lasso appropriately |
| 3    | Tree models | Can build/tune DecisionTree/RandomForest, explain overfitting control |
| 4    | Evaluation expertise | Selects appropriate metrics, uses CV correctly, diagnoses over/underfit |
| 5-6  | Pipeline completion | End-to-end project with preprocessing, modeling, evaluation, docs |

## 🔗 Integrated with Your GitHub Repo
Use your repo: https://github.com/Beastburner/MY_AI-ML_Journey

**Suggested Structure:**
```
MY_AI-ML_Journey/
├── README.md                 # Overall progress & goals
├── phase-1-classical-ml/
│   ├── week-1-2-linear-models/
│   │   ├── notes/
│   │   ├── exercises/
│   │   └── mini-projects/
│   ├── week-3-4-ensembles/
│   │   ├── notes/
│   │   ├── exercises/
│   │   └── mini-projects/
│   └── capstone-projects/
│       ├── house-price-prediction/
│       │   ├── data/
│   │   ├── notebooks/
│   │   ├── src/
│   │   └── README.md
│       └── titanic-survival/
│           ├── data/
│           ├── notebooks/
│           ├── src/
│           └── README.md
├── phase-2-math-stats/
├── phase-3-deep-learning/
├── ... (future phases)
└── resources/                # Helpful links, cheat sheets, etc.
```

**First Commit Message**: "Phase 1: Setting up study structure - Weeks 1-2 focus on linear models"

## ✅ Progress Update - Day 1 (April 15, 2026)

### What We Did Today:
1. ✅ **Data Loading & Exploration**: Loaded Sales_Train.csv (8,523 records, 12 features)
2. ✅ **EDA**: 
   - Created pairplot of numeric features
   - Generated correlation matrix heatmap (all weak correlations ~0)
   - Analyzed missing values and filled them appropriately
3. ✅ **Data Cleaning**:
   - Filled Item_Weight with mean (2,599 missing values)
   - Filled Outlet_Size using groupby mode by Outlet_Type + Outlet_Location_Type
4. ✅ **Model Building**:
   - Built Linear Regression model using scikit-learn
   - Used LabelEncoder for categorical variables
   - Split data 70/30 train/test
   - Made predictions on test set
5. ✅ **Model Evaluation**:
   - R² Score: 0.507 (explains 50.7% of variance)
   - RMSE: $1,174.78
   - Visualized actual vs predicted values (scatter plot)
6. ✅ **GitHub**: Created .gitignore and pushed initial commit

### What's Missing / Next Steps:

**Week 1 Remaining (Days 2-7):**
- [ ] Study linear regression theory (cost function, gradient descent, normal equation)
- [ ] Experiment with Ridge/Lasso regularization
- [ ] Learn feature scaling/normalization importance
- [ ] Create coefficient interpretation analysis
- [ ] Study logistic regression for classification
- [ ] Practice on multiple datasets (iris, diabetes, breast cancer)
- [ ] Create one-page linear models cheat sheet

**Immediate Improvements Needed:**
- [ ] Use OneHotEncoder instead of LabelEncoder for categorical features
- [ ] Implement StandardScaler for numeric features
- [ ] Try Ridge/Lasso regression to improve R²
- [ ] Experiment with better models (Random Forest, XGBoost)
- [ ] Feature engineering (create derived features)
- [ ] Cross-validation for robust evaluation

**Project Structure:**
- [x] Data loaded and explored
- [ ] Complete preprocessing pipeline
- [ ] Baseline model established (Linear Regression: R² 0.507)
- [ ] Feature engineering experiments
- [ ] Model comparison and selection
- [ ] Hyperparameter tuning
- [ ] Final evaluation and documentation

## 🚀 Immediate Action Items (Days 2-3)
1. **Theory**: Read scikit-learn Ridge/Lasso documentation + regularization explanation
2. **Practice**: Rebuild model with StandardScaler + OneHotEncoder
3. **Experiment**: Compare LinearRegression vs Ridge vs Lasso R² scores
4. **Learning**: Watch StatQuest "Linear Regression" video (30 mins)
5. **Documentation**: Update this file with results and learnings

## 📚 Resource Integration from Your Report
Use these specific resources mentioned in your deep-research-report.md:

**Primary Learning:**
- Scikit-learn user guide (train_test_split, pipelines, model API)【25†L1-L3】
- Andrew Ng’s “Machine Learning” specialization (Coursera) - regression, classification, decision trees, ensembles【33†L179-L187】
- Kaggle “Intro to Machine Learning” and “Advanced ML” micro-courses

**Hands-on Practice:**
- Kaggle Titanic/House Prices tutorials and example notebooks

**Theory/Reference:**
- *The Elements of Statistical Learning* (for in-depth algorithms)
- Breiman’s Random Forest paper (classic reference)

**YouTube Supplements:**
- *StatQuest* (Josh Starmer) for clear intuition on ML algorithms
- *3Blue1Brown* (Essence of linear algebra animations) - for upcoming Phase 2 math

**Remember from your report**: "You should complete simple projects (e.g. House Price Prediction (regression), Titanic Survival or Customer Churn (classification)) with ≥ 80% accuracy or reasonable error, to demonstrate mastery."

Would you like me to:
1. Create a similar detailed schedule for Week 1 specifically?
2. Help you set up the initial GitHub repo structure with starter files?
3. Suggest which project to start with (House Prices vs Titanic) based on your interests?
4. Create a daily log template for tracking your Phase 1 progress?