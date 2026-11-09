# 🧠 Student Mental Health Risk Prediction

> A structured, multi-feature classification dataset for predicting mental health risk levels among university students.

---

## 📋 Problem Statement

### Context
University students face a confluence of stressors — academic pressure, financial strain, social isolation, irregular sleep, and lifestyle disruptions — that place them at disproportionately high risk for mental health challenges. Despite growing awareness, early identification of at-risk students remains difficult due to the multifactorial nature of mental wellbeing.

### Objective
Build a machine learning model that predicts the **mental health risk level** (Low / Medium / High) of a university student based on behavioral, academic, lifestyle, and psychosocial features collected through a simulated student survey.

### Why It Matters
Mental health conditions affect approximately **1 in 3 university students** globally, yet most go undetected until they reach a crisis point. A predictive model could enable universities to proactively identify vulnerable students and offer early intervention — improving academic outcomes and student wellbeing simultaneously.

---

## 📁 Dataset Overview

| Property | Value |
|---|---|
| Rows | 2,000 |
| Columns | 17 (16 features + 1 target) |
| Task Type | Multi-class Classification |
| Target Variable | `Mental_Health_Risk` (Low / Medium / High) |
| Missing Values | Yes (~3–5% in selected columns) |

---

## 🗂️ Column Documentation

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `Student_ID` | string | Unique student identifier | STU0001 |
| `Age` | integer | Student age in years | 18–30 |
| `Gender` | categorical | Student gender identity | Male, Female, Non-binary |
| `Year_of_Study` | integer | Current academic year | 1–5 |
| `Sleep_Hours` | float | Average nightly sleep (hours) | 3.5–9.5 |
| `Study_Hours_Per_Week` | float | Self-reported study hours per week | 5–55 |
| `GPA` | float | Current cumulative GPA (4.0 scale) | 1.5–4.0 |
| `Social_Media_Hours` | float | Average daily social media usage (hours) | 0.5–10.0 |
| `Screen_Time_Hours` | float | Total daily screen time (hours) | 1.0–14.0 |
| `Physical_Activity` | categorical | Weekly exercise frequency level | Low, Moderate, High |
| `Financial_Stress` | float | Self-rated financial stress (1–10 scale) | 1.0–10.0 |
| `Family_Support` | float | Perceived family support level (1–10) | 1.0–10.0 |
| `Attendance_Rate` | float | Class attendance percentage | 30–100 |
| `Part_Time_Job` | binary | Whether student holds a part-time job | Yes, No |
| `Diet_Quality` | categorical | Self-reported diet quality | Poor, Average, Good |
| `Previous_Mental_Health_History` | binary | Prior diagnosed mental health condition | Yes, No |
| `Mental_Health_Risk` | categorical | **TARGET** — predicted risk level | Low, Medium, High |

---

## 📊 Feature Relationships & Data Logic

The target variable is derived from a **weighted risk score** using the following logical relationships:

- 🔴 **High risk indicators**: Low sleep (<6h), high financial stress (>7), prior mental health history, low GPA, high social media use, poor diet, part-time job
- 🟢 **Low risk indicators**: Adequate sleep (7–8h), high family support, active lifestyle, good diet, high attendance, strong GPA
- Risk is not purely deterministic — realistic noise is added to reflect real-world variability

Class distribution is roughly balanced: ~33% each (Low / Medium / High).

---

## 🔧 Data Quality & Preprocessing

### Missing Values
The following columns contain realistic missing data (~3–5%):

| Column | Missing % |
|---|---|
| Financial_Stress | ~4.5% |
| Diet_Quality | ~3.5% |
| Sleep_Hours | ~3.0% |
| Social_Media_Hours | ~3.0% |
| Family_Support | ~2.7% |
| Attendance_Rate | ~2.4% |
| GPA | ~1.9% |

### Recommended Preprocessing Steps
1. **Drop** `Student_ID` before modeling (identifier only)
2. **Impute** missing numerical values using median imputation or KNN imputation
3. **Impute** missing categorical values using mode imputation
4. **Encode** categorical features:
   - Ordinal encoding: `Physical_Activity`, `Diet_Quality`
   - One-hot encoding: `Gender`
   - Binary encoding: `Part_Time_Job`, `Previous_Mental_Health_History`
5. **Scale** numerical features using `StandardScaler` or `MinMaxScaler`
6. **Check for outliers** in `Study_Hours_Per_Week`, `Screen_Time_Hours` using IQR or Z-score

### Potential Outliers
- A small number of students report very high study hours (>50/week) or extreme sleep deprivation (<4h) — these are realistic edge cases and should be retained.

---

## 🤖 Baseline ML Approach

### Suggested Models

| Model | Rationale |
|---|---|
| **Logistic Regression** | Strong interpretable baseline; works well with scaled features |
| **Random Forest** | Handles non-linearity and mixed feature types; provides feature importance |
| **XGBoost** | State-of-the-art for tabular classification; robust to imbalance and noise |

### Evaluation Metrics
- **Primary**: Weighted F1-score (handles near-balanced multi-class well)
- **Secondary**: Accuracy, Macro F1, Confusion Matrix
- **Bonus**: ROC-AUC (one-vs-rest) for probability calibration assessment

### Suggested Splits
- Train/Validation/Test: 70% / 15% / 15%
- Use `StratifiedKFold` (k=5) for cross-validation to preserve class balance

---

## 💡 Suggested Analyses

- **Feature importance** plot using Random Forest / SHAP values
- **Correlation heatmap** of numeric features vs. risk level
- **Distribution plots**: sleep, GPA, financial stress by risk class
- **Sankey diagram**: Previous MH History × Physical Activity → Risk Level

---

## 📜 Citation & Usage

This is a **synthetic dataset** generated for educational and portfolio purposes. It simulates realistic patterns observed in student mental health research but does not represent any real individuals.

Feel free to use, adapt, and build upon this dataset. If you reference it, a link back to this Kaggle page is appreciated.

---

## 🏷️ Tags
`mental-health` · `student-data` · `classification` · `multiclass` · `lifestyle` · `education` · `psychology` · `synthetic-data` · `feature-engineering` · `random-forest`
