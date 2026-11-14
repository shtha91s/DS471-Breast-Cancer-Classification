# Breast Cancer Classification Using Machine Learning

## DS471 – Machine Learning Project

### Project Overview

This project focuses on classifying breast tumors as benign or malignant using machine learning techniques.

We use the Breast Cancer Wisconsin (Diagnostic) dataset, which contains numerical measurements extracted from digitized images of breast cell nuclei.

The main objective is to explore the dataset, prepare it for machine learning, and develop different classification models to predict breast cancer diagnosis.

### Dataset Information

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository

| Property | Description |
|---|---|
| Total Samples | 569 |
| Input Features | 30 |
| Benign Cases | 357 |
| Malignant Cases | 212 |
| Target Variable | Diagnosis |
| Classification Type | Binary Classification |

The target variable is encoded as:
- `0` – Benign
- `1` – Malignant

### Project Stages

#### Stage 1: Data Cleaning, Preparation, and Exploration

We performed the following tasks:

- Loaded and examined the dataset using Python.
- Checked for missing values and duplicate records.
- Identified potential outliers using the IQR method.
- Retained outliers because they may contain meaningful medical information.
- Analyzed the distribution of benign and malignant cases.
- Visualized selected numerical features.
- Examined feature correlations using a heatmap.
- Identified important features using Mutual Information calculated on training data only.

#### Stage 2: Data Splitting

The dataset was divided using stratified sampling:

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training | 398 | 70% |
| Validation | 85 | 15% |
| Testing | 86 | 15% |

A fixed random state of 42 was used for reproducibility.

#### Stage 3: Model Selection and Development

Four machine learning models were selected and configured:

1. **Logistic Regression:** An interpretable baseline for binary classification.
2. **Support Vector Machine (SVM):** Uses an RBF kernel to capture nonlinear relationships.
3. **Random Forest:** Combines multiple decision trees.
4. **Neural Network (MLP):** Uses one hidden layer with 32 neurons.

StandardScaler was included in the pipelines for Logistic Regression, SVM, and MLP.

### Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code

### Project Visualizations

The exploratory analysis includes:

- Outlier Detection Using Boxplots
- Diagnosis Distribution
- Feature Distributions by Diagnosis
- Correlation Heatmap
- Top 10 Features Based on Mutual Information

### How to Run the Project

1. Download or clone this repository.
2. Install the required Python libraries.
3. Open the project folder in Visual Studio Code.
4. Run the Python script.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the project:

```bash
python wdbc_project.py
```

The script generates exploratory analysis results and saves the figures in the `figures` folder.

### Current Project Status

- Stage 1: Completed
- Stage 2: Completed
- Stage 3: Models configured
- Stage 4: Training and hyperparameter optimization – Pending
- Stage 5: Final evaluation and comparison – Pending

### Academic Information

**Course:** DS471 – Machine Learning  
**Project:** Breast Cancer Classification  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)

