# DS471 – Breast Cancer Classification

## Project Overview
This project compares machine-learning models for classifying breast tumors as benign or malignant. It uses the Breast Cancer Wisconsin (Diagnostic) dataset.

## Dataset
The dataset contains 569 samples and 30 numerical features.

- Benign: 357 samples, encoded as `0`
- Malignant: 212 samples, encoded as `1`

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Project Files
- `DS471_Breast_Cancer_Classification_final (1).ipynb` — Notebook covering the project workflow.
- `wdbc_clean.csv` — Dataset used by the notebook.
- Project report — Final report submitted with the project.

## Models
The project develops and compares:
- Logistic Regression
- Support Vector Machine (SVM) with an RBF kernel
- Random Forest
- Multi-Layer Perceptron (MLP) Neural Network

## Project Stages
The notebook covers data preparation and exploration, stratified data splitting, model development, hyperparameter optimization, neural-network training and validation, and final test-set evaluation.

## Open in Google Colab
[Open the project notebook in Google Colab](https://colab.research.google.com/drive/1jm0oSefp_FSXegveQsle5Fw5GwqxcaiE)

## How to Run
1. Open the notebook using the Google Colab link above.
2. Upload `wdbc_clean.csv` when prompted.
3. Select **Runtime → Run all**.

The notebook uses Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
