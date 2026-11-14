import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()

# Create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Target encoding: 1 = Malignant, 0 = Benign
df["diagnosis"] = 1 - data.target

# Display the first five rows
print(df.head())

# Dataset information
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nTarget Distribution:")
print(df["diagnosis"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Stage 1 - Step 2: Outlier Detection

# Separate features from the target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Calculate IQR boundaries
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (X < lower_bound) | (X > upper_bound)

# Count outliers for each feature
outlier_counts = outliers.sum().sort_values(ascending=False)

print("\nOutlier Counts by Feature:")
print(outlier_counts)

# Count rows containing at least one outlier
rows_with_outliers = outliers.any(axis=1).sum()

print("\nRows with at least one outlier:", rows_with_outliers)

print("\nDecision: Outliers will be retained.")

# Stage 1 - Step 3: Outlier Visualization

import os

# Create a folder for project figures
os.makedirs("figures", exist_ok=True)

# Select features for visualization
selected_features = [
    "mean radius",
    "mean texture",
    "mean area",
    "mean concavity"
]

# Create boxplots
plt.figure(figsize=(12, 8))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)

    sns.boxplot(y=df[feature], color="skyblue")

    plt.title(feature)
    plt.ylabel("Value")

plt.suptitle("Outlier Detection Using Boxplots", fontsize=15)

plt.tight_layout()

# Save figure
plt.savefig(
    "figures/01_outlier_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFigure saved successfully!")

# Stage 1 - Step 4: Target Distribution

# Count benign and malignant cases
class_counts = df["diagnosis"].value_counts().sort_index()

print("\nDiagnosis Distribution:")
print("Benign:", class_counts[0])
print("Malignant:", class_counts[1])

# Create visualization
plt.figure(figsize=(8, 6))

ax = sns.countplot(
    data=df,
    x="diagnosis",
    order=[0, 1],
    hue="diagnosis",
    palette=["#4C78A8", "#E45756"],
    legend=False
)

plt.xticks([0, 1], ["Benign", "Malignant"])
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.title("Distribution of Benign and Malignant Cases")

# Display counts above bars
for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

# Save figure
plt.savefig(
    "figures/02_target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Stage 1 - Feature Distribution Analysis


selected_features = [
    "mean radius",
    "mean texture",
    "mean area",
    "mean concavity"
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, feature in zip(axes.flatten(), selected_features):

    sns.histplot(
        data=df,
        x=feature,
        hue="diagnosis",
        hue_order=[0, 1],
        bins=25,
        kde=True,
        stat="density",
        common_norm=False,
        palette={0: "#4C78A8", 1: "#E45756"},
        alpha=0.35,
        ax=ax
    )

    ax.set_title(feature.title())
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")

plt.suptitle(
    "Feature Distributions by Diagnosis",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "figures/03_feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Stage 1 - Step 6: Correlation Analysis


# Calculate correlation between numerical features
correlation_matrix = X.corr()

# Create correlation heatmap
plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.3
)

plt.title(
    "Correlation Heatmap of Breast Cancer Features",
    fontsize=16
)

plt.tight_layout()

# Save figure
plt.savefig(
    "figures/04_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Stage 2 - Data Splitting


from sklearn.model_selection import train_test_split

# Separate input features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Splitting Results:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nClass Distribution:")
print("Training:\n", y_train.value_counts())
print("Validation:\n", y_val.value_counts())
print("Testing:\n", y_test.value_counts())

# Stage 1 - Feature Importance
# Using Training Data Only


from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information scores
mi_scores = mutual_info_classif(
    X_train,
    y_train,
    random_state=42
)

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": mi_scores
})

# Sort features by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display the top 10 features
print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot the top 10 features
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature",
    color="steelblue"
)

plt.title("Top 10 Features Based on Mutual Information")
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")

plt.tight_layout()

# Save the figure
plt.savefig(
    "figures/05_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Stage 3 - Model Selection and Development


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Define the machine learning models
models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Neural Network": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            max_iter=1000,
            random_state=42
        ))
    ])
}

# Display selected models
print("\nStage 3 - Selected Models:")

for name, model in models.items():
    print(f"\n{name}")
    print(model)