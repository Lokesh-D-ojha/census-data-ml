# 📊 census-data-ml: Income Classification using U.S. Census Data

## Project Overview

This project focuses on **predicting whether an individual's income exceeds \$50K/year** based on U.S. Adult Census Data (often referred to as the **Adult dataset**). We perform extensive Exploratory Data Analysis (EDA), handle missing values using various imputation strategies, and apply machine learning models to solve this binary classification problem.

The primary goal is to build a robust model that can classify income levels accurately and to provide insights into the key demographic and employment factors that influence high earnings.

## 📁 Dataset

The project utilizes the `adult.csv` file. This dataset contains records extracted from the 1994 Census database and includes attributes like age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country, and the target variable, **income** (>$50K, <=$50K).

| Feature | Description |
| :--- | :--- |
| `age` | Age of the individual. |
| `workclass` | Type of employer (e.g., Private, Self-emp-not-inc). |
| `education` | The highest level of education achieved. |
| `occupation` | Type of job held. |
| `hours.per.week` | The number of hours worked per week. |
| `native.country` | Country of origin. |
| **`income`** | **Target variable**: Whether income is >$50K or <=$50K. |

## 🛠️ Technology Stack

* **Python** (3.8+)
* **Pandas** (Data manipulation and cleaning)
* **NumPy** (Numerical operations)
* **Scikit-learn** (Machine Learning models and utilities)
* **Matplotlib/Seaborn** (Data Visualization)

## 🔍 Exploratory Data Analysis (EDA) & Data Preprocessing

The EDA phase was crucial for understanding data quality, distributions, and relationships. Special attention was paid to handling missing values (`?` in the raw data) using context-specific imputation strategies.

### Missing Value Imputation Strategy

The following steps were taken to clean and impute missing values represented by the `?` character in the dataset:

1.  **Workclass Imputation:**
    * Missing values (`?`) in the `workclass` column were replaced with the **overall mode (most frequent value)** of the column.
    ```python
    frequentwc = df['workclass'].mode()[0]
    df['workclass'] = df['workclass'].replace('?', frequentwc)
    ```

2.  **Occupation Imputation (Conditional):**
    * Missing values in the `occupation` column were first converted to `NaN`.
    * Imputation was then performed **conditionally based on the `workclass`**. For each `workclass` category, the missing `occupation` values within that category were replaced by the **mode** of `occupation` for that specific `workclass`.
    * Any remaining missing values (due to rare or empty workclasses) were replaced with a placeholder: `'Unknown'`.
    ```python
    # Example logic used
    df['occupation'] = df['occupation'].replace('?', np.nan)
    for workclass in df['workclass'].unique():
        # ... conditional mode imputation logic ...
        df.loc[(df['workclass'] == workclass) & (df['occupation'].isna()), 'occupation'] = mode[0]
    df['occupation'] = df['occupation'].fillna('Unknown')
    ```

3.  **Native Country Imputation:**
    * Missing values (`?`) in the `native.country` column were first converted to `NaN`.
    * These missing values were then imputed with the **overall mode (most common country)**, which is typically 'United-States'.
    ```python
    df['native.country'] = df['native.country'].replace('?', np.nan)
    common_country = df['native.country'].mode()[0]
    df['native.country'] = df['native.country'].fillna(common_country)
    ```

---

## 🚀 Getting Started

### Prerequisites

Clone the repository and install the necessary Python packages:

```bash
git clone [ https://github.com/Lokesh-D-ojha/census-data-ml.git](https://github.com/YourUsername/census-data-ml.git)
cd census-data-ml
pip install -r requirements.txt
