# colorectal_cancer_prediction_test.ipynb
The goal of this project is to predict the survival status of colorectal cancer patients based on their demographics, medical history, and lifestyle factors.

# 🧬 Colorectal Cancer Risk & Survival Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Library-Pandas-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

## 📌 Executive Summary
Colorectal cancer is one of the leading causes of cancer-related deaths worldwide. Early prediction of survival outcomes can assist healthcare professionals in designing personalized treatment plans and optimizing medical interventions.

This project focuses on building an **end-to-end Machine Learning classification pipeline** to predict patient survival status (`Survived` vs `Deceased`) based on demographic details, clinical history, lifestyle parameters, and treatment adherence data across **89,945+ patient records**.

---

## 🛠️ Tech Stack & Libraries Used
* **Language:** Python 3.12
* **Data Processing & Manipulation:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning & Preprocessing:** Scikit-Learn (`LogisticRegression`, `StandardScaler`, `train_test_split`)
* **Model Metrics:** Classification Report, Accuracy Score, Confusion Matrix

---

## 📊 Dataset Highlights
* **Total Patient Records:** 89,945
* **Total Features:** 30 Clinical & Lifestyle Attributes
* **Target Variable:** `Survival_Status` (Binary Outcome: `Survived` / `Deceased`)
* **Class Distribution:**
  * **Survived:** 67,341 (~74.87%)
  * **Deceased:** 22,604 (~25.13%)

### Key Feature Categories:
1. **Demographics:** Age, Gender, Race, Region, Urban/Rural, Socioeconomic Status
2. **Medical History:** Family History, Previous Cancer, Diagnosis Stage (I - IV), Tumor Aggressiveness
3. **Lifestyle Factors:** BMI, Diet Type, Physical Activity Level, Smoking Status, Alcohol & Fiber Consumption
4. **Treatment & Access:** Colonoscopy Access, Insurance Coverage, Chemotherapy, Surgery, Radiotherapy, Recurrence Data

---

## 1. Data Cleaning & Exploration (EDA)
* Identified zero missing values across all features (`0 Null Values`).
* Evaluated target variable balance (75% Survived vs 25% Deceased).
* Explored feature correlations using heatmaps and numerical distributions.

### 2. Feature Engineering & Preprocessing
* Dropped non-informative identifier columns (`Patient_ID`).
* Cleaned target values and binary-encoded outcomes (`Survived = 1`, `Deceased = 0`).
* Converted multi-class categorical features using One-Hot Encoding (`pd.get_dummies`).
* Normalized continuous numerical variables (`Age`, `BMI`, `Time_to_Recurrence`) using `StandardScaler` to ensure uniform model convergence.

### 3. Model Building & Evaluation
* Split dataset into **80% Training Data** (71,956 samples) and **20% Testing Data** (17,989 samples) using stratified sampling.
* Trained a **Logistic Regression** baseline classifier.

#### Model Performance Results:
* **Accuracy Score:** `~74.87%`
* **Precision (Survived):** `0.75`
* **Recall (Survived):** `1.00`

---

## 📈 Key Insights & Healthcare Takeaways
1. **Dominant Class Baseline:** The vanilla Logistic Regression model achieved ~75% accuracy by identifying primary survival patterns across demographic factors.
2. **Class Imbalance Sensitivity:** Due to the 3:1 ratio between Survived and Deceased patients, advanced iterations can leverage cost-sensitive learning (`class_weight='balanced'`) or non-linear classifiers like **Random Forest** or **XGBoost** to boost recall for high-risk (Deceased) patients.

---

## 🚀 How to Run This Project Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/colorectal-cancer-survival-prediction.git](https://github.com/your-username/colorectal-cancer-survival-prediction.git)
cd colorectal-cancer-survival-prediction
