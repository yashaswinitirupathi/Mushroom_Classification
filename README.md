# Mushroom Classification using Machine Learning

A comparative machine learning study for classifying mushrooms as **edible or poisonous** using categorical morphological features. The project evaluates multiple classification algorithms and investigates how **feature scaling, outlier removal, PCA and LDA** affect model performance.

> **Academic Project | Machine Learning**

---

## Project Overview

Mushroom classification is a binary classification problem where the objective is to predict the class of a mushroom from its observable characteristics such as cap shape, odor, gill properties, stalk properties, habitat and population.

The project follows an end-to-end ML workflow:

**Data Loading → Pre-processing → EDA → Encoding → Train/Test Split → Outlier Analysis → Model Training → Dimensionality Reduction → Model Comparison → Evaluation**

The main focus is not only on achieving high accuracy, but also on understanding how different preprocessing and dimensionality-reduction strategies influence classification performance.

---

## Dataset

The dataset contains:

- **8,124 observations**
- **22 input features**
- **1 binary target variable (`class`)**
- No missing values
- Target classes:
  - `e` → Edible: **4,208**
  - `p` → Poisonous: **3,916**

All input variables are categorical. The notebook converts the categorical attributes into numerical representations using `OrdinalEncoder`.

### Feature Categories

The features describe characteristics including:

- Cap shape, surface and color
- Bruises
- Odor
- Gill attachment, spacing, size and color
- Stalk shape, root, surface and color
- Veil type and color
- Ring number and type
- Spore-print color
- Population
- Habitat

---

## Objectives

1. Explore and understand the structure of the dataset.
2. Pre-process categorical features for machine learning.
3. Analyze class distribution, correlations, duplicates and potential outliers.
4. Establish baseline classification models.
5. Compare:
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Decision Tree
6. Study the impact of **outlier removal**.
7. Apply **Principal Component Analysis (PCA)** for dimensionality reduction.
8. Apply **Linear Discriminant Analysis (LDA)** for supervised dimensionality reduction.
9. Evaluate models using accuracy, precision, recall, F1-score and confusion matrices.

---

## Methodology

### 1. Data Pre-processing

- Loaded the dataset using Pandas.
- Checked for missing values.
- Identified categorical variables.
- Applied `OrdinalEncoder` to convert categorical attributes into numerical form.
- Examined the target-class distribution.

### 2. Exploratory Data Analysis

The notebook performs:

- Dataset summary and descriptive statistics
- Duplicate-row analysis
- Feature distribution analysis
- Correlation analysis
- Outlier identification using the **IQR method**
- Box-plot based visualization of feature distributions

Using the notebook's IQR-based procedure after ordinal encoding, approximately **4.19% of all encoded cells** were flagged as outliers. Removing any row containing an identified outlier retained **3,600 observations**.

> Note: because the categorical variables were ordinal-encoded before applying IQR, these outliers represent the notebook's preprocessing-based statistical definition rather than continuous physical measurements.

### 3. Train/Test Split

The original dataset was split using:

- **70% training data**
- **30% test data**
- `random_state = 42`

### 4. Dimensionality Reduction

Two approaches were investigated:

#### PCA — Principal Component Analysis

The features were standardized before PCA.

- **10 principal components** were retained.
- On the original data, these 10 components explained approximately **83.69%** of the variance.
- After outlier removal, they explained approximately **93.82%** of the variance.

#### LDA — Linear Discriminant Analysis

LDA was used as a supervised dimensionality-reduction technique with the binary class labels.

---

## Models Evaluated

### K-Nearest Neighbors

Cross-validation was used to investigate the choice of `k` from 1 to 20.

The notebook identified:

**Best k = 1**

with a mean 10-fold cross-validated training accuracy of approximately **99.96%**.

### Logistic Regression

Used with `max_iter=1000` to ensure convergence.

### Support Vector Machine

Evaluated using `SVC`; a linear kernel was also examined for the LDA-based experiments.

### Decision Tree

Decision trees were evaluated using both:

- Gini impurity
- Entropy

The trees were also restricted to `max_depth=3` in the initial experiments.

---

## Results

The notebook reports the following test-set accuracies for the main experiments:

| Approach | Model | Test Accuracy |
|---|---|---:|
| Original features | KNN (k=1) | **99.88%** |
| Original features | KNN (k=5) | 99.55% |
| Original features | KNN (k=10) | 99.26% |
| Original features | Logistic Regression | 94.91% |
| Original features | SVM | **100.00%** |
| Original features | Decision Tree – Gini | 95.78% |
| Original features | Decision Tree – Entropy | 95.32% |
| PCA + outliers | KNN | 99.84% |
| PCA + outliers | Logistic Regression | 90.24% |
| PCA + outliers | SVM | **99.84%** |
| PCA + outliers | Decision Tree | 87.57% |
| PCA + no outliers | KNN | **100.00%** |
| PCA + no outliers | Logistic Regression | 99.07% |
| PCA + no outliers | SVM | **100.00%** |
| PCA + no outliers | Decision Tree | 98.89% |
| LDA + outliers | KNN | 92.74% |
| LDA + outliers | Logistic Regression | 94.95% |
| LDA + outliers | SVM | 95.04% |
| LDA + outliers | Decision Tree | 92.74% |
| LDA + no outliers | KNN | 99.35% |
| LDA + no outliers | Logistic Regression | 99.35% |
| LDA + no outliers | SVM | **100.00%** |
| LDA + no outliers | Decision Tree | **100.00%** |

### Key Observations

- KNN achieved very high performance even without dimensionality reduction.
- SVM achieved **100% test accuracy** on the original standardized feature representation in the notebook.
- Removing observations identified as outliers substantially improved the performance of several models.
- PCA reduced the feature space while retaining a large proportion of the variance.
- LDA performed considerably better after the outlier-removal step.
- Model performance varied substantially depending on the preprocessing and representation of the data, demonstrating the importance of the complete ML pipeline rather than model selection alone.

---

## Evaluation Metrics

Model performance was assessed using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

Confusion matrices were visualized using Seaborn to examine class-specific prediction errors.

---

## Tech Stack

**Programming**
- Python

**Libraries**
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Graphviz

**ML Techniques**
- KNN
- Logistic Regression
- SVM
- Decision Trees
- PCA
- LDA
- Cross-validation
- Outlier detection using IQR

---

## Repository Structure

```text
Mushroom-Classification/
│
├── Mushroom_Classification_project_.ipynb
├── mushrooms.csv
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-link>
cd Mushroom-Classification
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn graphviz jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook Mushroom_Classification_project_.ipynb
```

### 4. Dataset path

The notebook was originally developed in a Google Colab environment and reads:

```python
pd.read_csv("/content/mushrooms.csv")
```

For local execution, change the path to:

```python
pd.read_csv("mushrooms.csv")
```

---

## Project Highlights

- Performed complete **EDA and preprocessing** on 8,124 observations.
- Compared **four classification algorithms** under multiple preprocessing conditions.
- Used **10-fold cross-validation** to investigate the optimal K value for KNN.
- Studied the effect of **outlier removal on classification performance**.
- Applied both **PCA and LDA** for dimensionality reduction.
- Evaluated models using multiple classification metrics and confusion matrices.
- Demonstrated how preprocessing choices can significantly influence ML model performance.

---



