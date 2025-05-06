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















