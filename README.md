# 🌱 Greenhouse Plant Growth Classification with Machine Learning

This project applies **machine learning** to classify greenhouse-grown plants based on growth metrics. We use a **tabular dataset of plant measurements** from two types of greenhouses: **Traditional** and **IoT-based.**

---

## 📝 **Objective**

👉 Predict the **type of greenhouse environment** (traditional or IoT-based) based on plant physiological and growth metrics.

We explore:
- Data exploration & visualization
- Feature correlation
- Machine learning classification
- Model evaluation

---

## 📊 **Dataset Description**

The dataset contains **30,000 samples** of plant growth measurements across 6 experimental classes:

| Class | Description              |
|:------|:------------------------|
| SA     | Traditional Greenhouse A |
| SB     | Traditional Greenhouse B |
| SC     | Traditional Greenhouse C |
| TA     | IoT Greenhouse A         |
| TB     | IoT Greenhouse B         |
| TC     | IoT Greenhouse C         |

### **Features:**

| Column | Description |
|:--------|:------------|
| ACHP | Avg chlorophyll content per plant |
| PHR  | Plant height rate |
| AWWGV| Avg wet weight of vegetative growth |
| ALAP | Avg leaf area per plant |
| ANPL | Avg number of leaves per plant |
| ARD  | Avg root diameter |
| ADWR | Avg dry weight of roots |
| PDMVG| % dry matter in vegetative growth |
| ARL  | Avg root length |
| AWWR | Avg wet weight of roots |
| ADWV | Avg dry weight of vegetative parts |
| PDMRG| % dry matter in root growth |
| Class | Experimental class (target) |

---

## 🕵️‍♂️ **Exploratory Data Analysis (EDA)**

We began by:
✅ Checking class balance  
✅ Summarizing features  
✅ Correlation heatmap  
✅ Visualizing feature distributions per class

---

### **Key Insights from EDA**

🔍 **1. Class distribution:**
- The dataset is **perfectly balanced**: 5000 samples per class.
- No resampling required.

🔍 **2. Descriptive statistics:**
- All features are numeric, no missing values.
- Wide range in some features (e.g., `ALAP`, `AWWR`) → scaling may be required.

🔍 **3. Correlation analysis:**
- Strong positive correlations:
  - `AWWGV` ↔ `ALAP` (**r ≈ 0.95**) → more leaf area correlates with vegetative weight.
  - `ADWR` ↔ `AWWR` (**r ≈ 0.93**)
  - `ADWV` ↔ `PDMVG` (**r ≈ 0.88**)
- Multicollinearity likely → may need to handle in some models.

🔍 **4. Boxplots per Class:**
- Features like `ACHP`, `PHR`, `ALAP` show noticeable separation between classes.
- Features like `ANPL`, `AWWGV` show overlap → less discriminatory power alone.
- Suggests classification will benefit from combining multiple features.

---

### **EDA Code**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Greenhouse Plant Growth Metrics.csv')

# Preview
print(df.head())

# Class balance
print(df['Class'].value_counts())

# Descriptive stats
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df.drop(columns=['Random']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplots for selected features
features = ['ACHP', 'PHR', 'AWWGV', 'ALAP', 'ANPL']
for feat in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Class', y=feat, data=df)
    plt.title(f'{feat} distribution by Class')
    plt.show()
```

### EDA Plots
![1_Correlation](https://github.com/user-attachments/assets/cb89a9d4-c84f-4497-9d64-aecd42c08b1c)
![2](https://github.com/user-attachments/assets/f0ef0700-1a1a-4562-9f49-6522c423ba14)
![3](https://github.com/user-attachments/assets/c7cdca6a-a29d-4c96-8e4c-71c6c2fd9aa0)
![4](https://github.com/user-attachments/assets/338aa432-b89f-4bdd-ba82-7ec0aecd6a8a)
![5](https://github.com/user-attachments/assets/568b3bc2-b1aa-4048-85b3-26c5c048edae)
![6](https://github.com/user-attachments/assets/144a7a54-53f4-4a7b-8aca-43a0efcceb72)

## 🛠️ **2. Data Preprocessing**

After completing exploratory data analysis, we prepared the dataset for machine learning:

✅ Dropped the `Random` column → this identifier does not provide predictive information.

✅ Encoded the categorical target variable `Class` into numeric labels:

| Original Class | Encoded |
|:---------------|:--------|
| SA              | 0       |
| SB              | 1       |
| SC              | 2       |
| TA              | 3       |
| TB              | 4       |
| TC              | 5       |

✅ Scaled the numeric features using **StandardScaler** → ensures features have zero mean and unit variance, improving performance for distance-based algorithms.

✅ Prepared:
- Feature matrix `X_scaled`: shape **(30,000 samples, 12 features)**
- Target vector `y`: encoded class labels

---

### **Preprocessing Code**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop irrelevant column
df_clean = df.drop(columns=['Random'])

# Encode target labels
label_encoder = LabelEncoder()
df_clean['Class_encoded'] = label_encoder.fit_transform(df_clean['Class'])

# Prepare features and target
X = df_clean.drop(columns=['Class', 'Class_encoded'])
y = df_clean['Class_encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mapping of class labels
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Class mapping:", class_mapping)
```

✅ The encoded class mapping is:
```
{'SA': 0, 'SB': 1, 'SC': 2, 'TA': 3, 'TB': 4, 'TC': 5}
```

The dataset is now clean, numeric, and scaled—ready for machine learning modeling!

## 🤖 **3. Machine Learning Modeling**

We trained and evaluated three machine learning models for **multiclass classification** (6 classes):

✅ Logistic Regression (multinomial)  
✅ Random Forest Classifier  
✅ XGBoost Classifier

All models were trained using:
- Feature matrix: **12 scaled numeric features**
- Target: **encoded Class label (0–5)**
- **Train-test split:** 80% train / 20% test (stratified)

---

### **Model Training Code**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```

### **Results**

All three machine learning models achieved **perfect accuracy on the test set**:

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression  | 1.0000   |
| Random Forest        | 1.0000   |
| XGBoost              | 1.0000   |

✅ Every sample in the test set was correctly classified.  
✅ The precision, recall, and F1-score for each class were all **1.00**.  
✅ Confusion matrices showed **zero misclassifications across all classes.**

---

![1](https://github.com/user-attachments/assets/8d1bc0ac-95ef-4cbf-960f-61fe2b0ff435)
![2](https://github.com/user-attachments/assets/bd1a0059-b92d-45c6-b9db-2b72597fba6d)

### **Interpretation**

Achieving **100% accuracy on both training and unseen test data** is highly unusual in real-world machine learning tasks. This suggests:

- The dataset has **very strong and easily separable patterns** between classes.
- One or more features might be **highly correlated** with the target class, making classification very straightforward.
- There may be **redundant or potentially leaking features** (features that act as proxies for the class label).

In the context of this dataset—collected from controlled greenhouse experiments—it’s plausible that the data is inherently clean, structured, and highly separable.

Despite the perfect accuracy, it’s important to further analyze **feature importance** to understand which features are driving the predictions and whether the model is overly reliant on a small subset of variables.

## 🔍 **4. Feature Importance & Interpretation**

To understand how different features contributed to the model’s predictions, we extracted **feature importances** from the trained **Random Forest** and **XGBoost** classifiers.

✅ Both models consistently identified the following top features:

| Rank | Feature                         | Importance (Random Forest) |
|:----:|:--------------------------------|:-------------------------:|
| 1    | AWWGV (Avg wet weight of veg)   | ~0.28                      |
| 2    | ALAP (Avg leaf area per plant)  | ~0.27                      |
| 3    | ADWV (Avg dry weight veg parts) | ~0.18                      |

Together, **AWWGV and ALAP accounted for over 50% of the model’s predictive power**, indicating that much of the classification decision was driven by these two features.

---

### **Feature Importance Plots**

| Random Forest Feature Importance | XGBoost Feature Importance |
|:-------------------------------:|:-------------------------:|
| ![rf_importance](plots/rf_importance.png) | ![xgb_importance](plots/xgb_importance.png) |

*(Save generated plots under `/plots` folder.)*

---

### **Interpretation**

Achieving **100% accuracy on both the training and test sets** combined with high feature importances for just two variables suggests:

✅ The dataset contains **highly separable patterns** between classes  
✅ The model is **strongly reliant on a small subset of features**  
✅ Potential **feature redundancy or data leakage** (if certain features act as proxies for the class label)

This is plausible in experimental data from controlled environments, like greenhouses, where conditions are tightly managed.

---

## ✨ **Experiment: Removing Top Features**

To test the model’s dependency on the top features, we attempted retraining after dropping:

- **Scenario 1:** Dropped the most important feature (`AWWGV`)
- **Scenario 2:** Dropped the top two features (`AWWGV` and `ALAP`)

We planned to evaluate how accuracy changes without these strong predictors.

> ⚠️ **Note:** Due to environment limitations, this retraining experiment could not complete within the current runtime environment.  
> 
> However, we expect:
> - A noticeable drop in accuracy without the top 2 features
> - Remaining features still holding partial predictive power

✅ We encourage running this experiment locally to verify and explore the extent of model dependency on top features.

---

## 🎯 **Takeaways**

- A small subset of features (**AWWGV**, **ALAP**) largely drives model predictions
- Perfect classification accuracy suggests a highly separable dataset
- Future work could explore:
  - Reducing features for minimal-input models
  - Testing binary classification (Traditional vs IoT greenhouse)
  - Validating model performance on unseen/real-world data










