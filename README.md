# ICU Admission Prediction for Covid-19 Patients

## Introduction

The Covid-19 pandemic placed significant strain on healthcare facilities worldwide, particularly on Intensive Care Unit (ICU) resources. Understanding the factors that influence whether a patient requires ICU care is crucial for early intervention and effective resource planning.

This study analyzes a large-scale Covid-19 dataset from the Government of Mexico, sourced from Kaggle:
[https://www.kaggle.com/datasets/meirnizri/covid19-dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)

The original dataset contained **1,048,575 patient records**. After extensive data cleaning to remove inconsistencies, address missing values, and retain all relevant variables for analysis, the refined dataset includes **1,019,666 individuals**. Each record consists of **21 clinical and demographic variables**, such as age, pneumonia status, diabetes, and cardiovascular conditions.

The key target variable indicates whether a patient was admitted to the ICU, forming the basis for predictive modeling and comparative analysis.

---

## Dataset Description

The dataset comprises **21 individual-level health variables**, including:

* **Demographic variables**

  * `SEX` (1 = Female, 2 = Male)
  * `AGE`

* **Clinical and diagnostic variables**

  * `CLASSIFICATION_FINAL` (1–3 = Covid-19 positive, ≥4 = negative or inconclusive)
  * `PATIENT_TYPE` (1 = Returned home, 2 = Hospitalized)
  * `PNEUMONIA`
  * `INTUBED`
  * `ICU`
  * `DATE_DIED`

* **Comorbidities** (1 = Yes, 2 = No)

  * Diabetes
  * COPD
  * Asthma
  * Hypertension
  * Cardiovascular disease
  * Renal chronic disease
  * Obesity
  * Other disease

* **Additional variables**

  * `PREGNANT`
  * `TOBACCO`
  * `INMSUPR` (Immunosuppression)
  * `USMER`
  * `MEDICAL_UNIT`

---

## Data Cleaning and Preprocessing

Several variables required targeted modifications to prevent loss of meaningful data:

* **Pregnancy**: Applicable only to females. Male records were assigned `2` (not pregnant) instead of missing values.
* **Intubated and ICU**: Non-hospitalized patients had placeholder values (97, 98, 99), which were recoded as `2` (No).
* **Date Died**: The placeholder `9999-99-99` was converted into a binary survival indicator.
* **Age**: Retained as-is to avoid removing individuals with values such as 97–99, which may represent real ages.

These steps ensured no variable collapsed into a single constant value and preserved the dataset’s integrity for modeling.

---

## Exploratory Data Analysis (EDA)

### Figure 1. Principal Component Analysis (PCA)

Principal Component Analysis (PCA) was used to reduce dimensionality and visualize underlying structure. The first two principal components capture most of the variance, forming a dense, elliptical distribution primarily aligned with the first component.

The absence of strong clustering suggests no clear separations in the reduced space, though gradual gradients indicate latent structure that may be revealed through supervised learning or nonlinear techniques.

---

### Figure 2. Histograms of All Variables

Most variables are binary and exhibit strong class imbalance. Features such as `INTUBED`, `PREGNANT`, `ICU`, and several comorbidities are dominated by a single category, reflecting their relative rarity in the population.

* `AGE` shows a right-skewed distribution centered between 20 and 60 years.
* `CLASSIFICATION_FINAL` indicates most patients were Covid-19 positive.
* `MEDICAL_UNIT` and `USMER` display uneven distributions, suggesting concentration in certain healthcare facilities.

---

### Figure 3. Boxplots and Summary Statistics

Boxplots reveal minimal variation for most categorical variables due to their binary nature.

* `AGE` shows moderate skewness with a wide interquartile range and high variance.
* `MEDICAL_UNIT` exhibits substantial spread, highlighting heterogeneity in healthcare facility usage.
* Critical care variables (`ICU`, `INTUBED`, `DATE_DIED`) show strong imbalance, consistent with earlier findings.

Overall, most comorbidities have means near `2` and low variance, indicating that the majority of patients did not present with these conditions.

---

### Figure 4. Correlation Matrix

The correlation analysis reveals:

* Moderate positive correlations among comorbidities such as diabetes, hypertension, renal disease, and immunosuppression.
* `AGE` correlates positively with pneumonia, patient type, and intubation, emphasizing its role in disease severity.
* `INTUBED` correlates with ICU admission and mortality, reinforcing its association with severe outcomes.

Several variables (e.g., sex, pregnancy, tobacco use) show negligible linear correlations, suggesting their influence may be nonlinear or context-dependent.

---

## Model Performance and Evaluation

### Figure 5. Logistic Regression Confusion Matrix

* **Accuracy**: 58.4%
* **Sensitivity**: 99.73%
* **Specificity**: 17.07%

The model strongly favors ICU prediction, leading to many false positives among non-ICU patients.

---

### Figure 6. Support Vector Machine (SVM)

* **Accuracy**: 53.02%
* **Sensitivity**: 99.93%
* **Specificity**: 6.12%

The SVM almost always predicts ICU admission, resulting in extremely low specificity due to severe class imbalance.

---

### Figure 7. Decision Tree

* **Accuracy**: 67.57%
* **Sensitivity**: 98.92%
* **Specificity**: 36.22%

The Decision Tree offers the best balance, improving non-ICU classification while maintaining high sensitivity.

---

### Figure 8. Zero-Hidden-Layer Neural Network

* **Accuracy**: 56.9%
* **Sensitivity**: 99.85%
* **Specificity**: 13.94%

This linear classifier favors recall over precision, highlighting the limitations of shallow architectures for complex clinical data.

---

## Table 1. Model Performance Comparison

| Model               | Accuracy | Sensitivity | Specificity |
| ------------------- | -------- | ----------- | ----------- |
| Logistic Regression | 0.5840   | 0.9973      | 0.1707      |
| SVM                 | 0.5302   | 0.9993      | 0.0612      |
| Decision Tree       | 0.6757   | 0.9892      | 0.3622      |
| Neural Network      | 0.5690   | 0.9985      | 0.1394      |

---

## Conclusion

All models demonstrated very high sensitivity, making them effective at identifying patients requiring ICU care. However, this often came at the cost of poor specificity.

The **Decision Tree** emerged as the most reliable model, achieving the highest accuracy and specificity while maintaining strong sensitivity. Its balanced performance makes it the most suitable choice for ICU admission prediction in this dataset.

Future work may explore ensemble methods, class rebalancing, or deeper neural architectures to further improve discrimination between ICU and non-ICU patients.
