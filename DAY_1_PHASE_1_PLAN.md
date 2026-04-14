# Day 1: Phase 1 - Classical Machine Learning Kickoff
## Your First Step into ML Engineering

**Goal**: Establish your ML workflow foundation and complete your first scikit-learn exercise

**Time Allocation**: 2.5-3 hours focused work
**Best Time**: Morning when your mind is freshest for new concepts

## 🎯 Day 1 Objectives
By end of day, you will be able to:
1. Set up your ML environment (conda, jupyterlab)
2. Load and explore a dataset using pandas
3. Split data into train/test sets using scikit-learn
4. Train your first model (LogisticRegression)
5. Evaluate model performance
6. Document your process in your GitHub repo

## 📅 Detailed Schedule

### **Hour 1: Environment Setup & GitHub Preparation**
**(25-30 minutes)**

**Minutes 0-10: Environment Verification**
- Open Anaconda Prompt (or terminal)
- Check if conda is available: `conda --version`
- Create ML environment: `conda create -n ml-roadmap python=3.9 -y`
- Activate environment: `conda activate ml-roadmap`
- Install essential packages: 
  ```
  conda install numpy pandas scikit-learn matplotlib seaborn jupyterlab -y
  ```
  *or*
  ```
  pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab
  ```

**Minutes 10-20: GitHub Repository Setup**
- Go to: https://github.com/Beastburner/MY_AI-ML_Journey
- Ensure you're signed in
- Create these directories via GitHub web interface OR locally:
  ```
  MY_AI-ML_Journey/
  ├── phase-1-classical-ml/
  │   └── week-1-2-linear-models/
  │       ├── day-1-exercises/
  │       └── notes/
  ├── resources/
  └── logs/
  ```
- **First Commit**: Add empty README.md in phase-1 folder with commit message:
  `"Phase 1: Setting up directory structure for classical ML studies"`

**Minutes 20-25: JupyterLab Launch**
- Start JupyterLab: `jupyter lab`
- Keep it running in background/tmux/screen
- Create new notebook: `day-1-ml-kickoff.ipynb` in `phase-1-classical-ml/week-1-2-linear-models/day-1-exercises/`

### **Hour 2: Hands-on ML Practice**
**(50-60 minutes)**

**Minutes 25-35: Dataset Loading & Exploration**
In your notebook, execute:
```python
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the iris dataset (built into scikit-learn - perfect for first exercise)
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Quick exploration
print("Dataset shape:", iris_df.shape)
print("\nFirst 5 rows:")
print(iris_df.head())
print("\nSpecies distribution:")
print(iris_df['species_name'].value_counts())
```

**Minutes 35-45: Data Preparation & Splitting**
Continue in notebook:
```python
# Prepare features and target
X = iris_df[iris.feature_names]  # All 4 features
y = iris_df['species']           # Target (0, 1, 2)

# Split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("\nClass distribution in training:")
print(pd.Series(y_train).value_counts(normalize=True))
```

**Minutes 45-55: Model Training**
Continue in notebook:
```python
# Initialize and train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Model trained successfully!")
print("Training accuracy:", accuracy_score(y_train, y_pred_train))
print("Test accuracy:", accuracy_score(y_test, y_pred_test))
```

**Minutes 55-65: Evaluation & Interpretation**
Continue in notebook:
```python
# Detailed classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, 
                          target_names=iris.target_names))

# Feature importance (coefficients for each class)
feature_importance = pd.DataFrame(
    model.coef_.T,
    columns=['setosa', 'versicolor', 'virginica'],
    index=iris.feature_names
)
print("\nFeature Coefficients:")
print(feature_importance)

# Visualize feature importance
feature_importance.plot(kind='bar')
plt.title('Logistic Regression Feature Coefficients by Class')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### **Hour 3: Documentation & Reflection**
**(30-40 minutes)**

**Minutes 65-80: GitHub Documentation**
1. Save your notebook
2. Create a README.md in `day-1-exercises/` folder with:
   ```
   # Day 1: ML Kickoff Exercise
   
   ## What I Did
   - Set up conda environment for ML roadmap
   - Loaded and explored the iris dataset
   - Split data into train/test sets (80/20)
   - Trained a LogisticRegression classifier
   - Evaluated model performance
   
   ## Key Learnings
   - Train/test split prevents overfitting optimism
   - LogisticRegression gives probabilities for each class
   - Feature coefficients show which variables influence predictions most
   - 80/20 split is a good starting point for model evaluation
   
   ## Questions for Tomorrow
   - How do I choose between different scalers (StandardScaler vs MinMaxScaler)?
   - When should I use stratification in train/test split?
   - What's the difference between accuracy and F1-score?
   ```
3. Commit and push your changes:
   ```
   git add .
   git commit -m "Day 1: Completed first ML exercise with iris dataset - environment setup and LogisticRegression basics"
   git push origin main
   ```

**Minutes 80-90: Reflection & Planning**
- Take 5 minutes to write in your learning log:
  - What felt easy? What felt confusing?
  - One "aha!" moment from today
  - What you're most excited to learn tomorrow
- Plan tomorrow's focus: 
  - Option A: Practice with different datasets (wine, breast cancer)
  - Option B: Explore preprocessing (scaling, encoding)
  - Option C: Try different models (Decision Tree, SVM)

## ✅ Success Criteria for Day 1
You've succeeded if you can:
- [ ] Activate your conda ML environment without help
- [ ] Load a dataset and basic exploration with pandas
- [ ] Split data using train_test_split with proper parameters
- [ ] Train and evaluate a LogisticRegression model
- [ ] Push working code to your GitHub repo with clear commit message
- [ ] Document what you learned and questions you have

## 📚 Resources Used Today
- **scikit-learn iris dataset**: Perfect starter dataset - clean, well-documented, multiclass
- **Train/test split**: Fundamental technique to prevent overfitting (from your report)
- **LogisticRegression**: First classification model - interpretable and fast
- **Classification report**: Shows precision/recall/F1 - better than accuracy alone

## 🔗 Connections to Your Roadmap
Today's work directly supports:
- **Phase 1 Objective**: "Learn to take raw data → preprocess → train & evaluate a model"
- **Tools**: scikit-learn API (fit(), predict(), train_test_split)
- **Metrics**: Accuracy, precision, recall, F1-score
- **Mindset**: Build, don't just watch - you coded along!

## 🚀 Tomorrow's Preview
Based on your energy and interests tomorrow, you could:
1. **Try a regression problem**: Use the diabetes dataset to predict quantitative progression
2. **Explore preprocessing**: Add StandardScaler and see how it affects model performance
3. **Compare models**: Train Decision Tree alongside Logistic Regression
4. **Deeper dive**: Look at the coefficients and what they mean for feature importance

Would you like me to:
1. Create a template for your daily learning log?
2. Suggest specific exercises for Day 2 based on how Day 1 went?
3. Help you set up a tracking system for your Phase 1 progress metrics?
4. Share some common beginner pitfalls to avoid in ML?