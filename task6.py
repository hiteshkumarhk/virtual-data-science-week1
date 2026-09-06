# ============================================================
# WEEK 6 - INTEGRATIVE DATA SCIENCE CAPSTONE PROJECT
# TITANIC SURVIVAL PREDICTION
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

DATA_PATH = "titanic_cleaned_week1.csv"

OUTPUT_DIR = "week6_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("WEEK 6 - DATA SCIENCE CAPSTONE PROJECT")
print("Titanic Survival Prediction")
print("=" * 70)


# ============================================================
# 2. DATA COLLECTION / LOADING
# ============================================================

print("\n[1] Loading Dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place titanic_cleaned_week1.csv in the same folder as this script."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. BASIC DATA UNDERSTANDING
# ============================================================

print("\n[2] Basic Data Understanding")

print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# Detect target column
possible_targets = [
    "survived",
    "Survived",
    "target",
    "Target"
]

target = None

for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    raise ValueError("Target column 'survived' was not found.")

print("\nTarget column:", target)

print("\nTarget distribution:")
print(df[target].value_counts())

print("\nTarget percentage:")
print(df[target].value_counts(normalize=True) * 100)


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\n[3] Data Cleaning")

df_clean = df.copy()

# Remove duplicate rows
duplicate_count = df_clean.duplicated().sum()

print("Duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df_clean = df_clean.drop_duplicates()

print("Shape after duplicate removal:", df_clean.shape)


# ============================================================
# 5. TARGET LEAKAGE CHECK
# ============================================================

print("\n[4] Target Leakage Check")

if "alive" in df_clean.columns:

    print("\n'alive' column detected.")

    print("Alive value counts:")
    print(df_clean["alive"].value_counts())

    print("\nRelationship between alive and survived:")

    print(
        pd.crosstab(
            df_clean["alive"],
            df_clean[target]
        )
    )

    print(
        "\nIMPORTANT:"
        "\n'alive' directly represents the survival outcome."
        "\nTherefore it MUST NOT be used as a predictor."
    )

    df_model = df_clean.drop(columns=["alive"])

else:

    print("'alive' column not found.")
    df_model = df_clean.copy()


# ============================================================
# 6. REMOVE IDENTIFIER / HIGH-CARDINALITY COLUMNS
# ============================================================

print("\n[5] Removing Unnecessary Columns")

columns_to_drop = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

existing_drop_columns = [
    col for col in columns_to_drop
    if col in df_model.columns
]

print("Columns removed:")
print(existing_drop_columns)

df_model = df_model.drop(
    columns=existing_drop_columns,
    errors="ignore"
)

print("\nRemaining columns:")
print(df_model.columns.tolist())


# ============================================================
# 7. FEATURE / TARGET SEPARATION
# ============================================================

print("\n[6] Preparing Features and Target")

X = df_model.drop(columns=[target])

y = df_model[target].astype(int)

print("Feature shape:", X.shape)

print("Target shape:", y.shape)


# ============================================================
# 8. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number", "bool"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 9. PREPROCESSING PIPELINE
# ============================================================

print("\n[7] Creating Preprocessing Pipeline")

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 10. TRAIN TEST SPLIT
# ============================================================

print("\n[8] Splitting Dataset")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 11. CREATE MODELS
# ============================================================

print("\n[9] Creating Machine Learning Models")

models = {

    "Logistic Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    max_depth=8,
                    min_samples_split=5
                )
            )
        ]
    )
}


# ============================================================
# 12. MODEL TRAINING
# ============================================================

print("\n[10] Training Models")

results = []

predictions = {}

probabilities = {}

for model_name, model in models.items():

    print("\nTraining:", model_name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[model_name] = y_pred

    probabilities[model_name] = y_prob

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc
        }
    )


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1", "ROC-AUC"],
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]

best_predictions = predictions[
    best_model_name
]

best_probabilities = probabilities[
    best_model_name
]

print("\nBest Model:", best_model_name)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n[11] Classification Report")

report = classification_report(
    y_test,
    best_predictions,
    zero_division=0
)

print(report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write(report)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\n[12] Creating Confusion Matrix")

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did Not Survive", "Survived"]
)

disp.plot()

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. ROC CURVES
# ============================================================

print("\n[13] Creating ROC Curve")

plt.figure(figsize=(8, 6))

for model_name in models:

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities[model_name]
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities[model_name]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_score:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 18. MODEL COMPARISON GRAPH
# ============================================================

print("\n[14] Creating Model Comparison Chart")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

results_plot = results_df.set_index(
    "Model"
)[metrics]

results_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 19. CROSS VALIDATION
# ============================================================

print("\n[15] Performing 5-Fold Cross Validation")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for model_name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    cv_results.append(
        {
            "Model": model_name,
            "CV Accuracy Mean": scores.mean(),
            "CV Accuracy Std": scores.std(),
            "Fold 1": scores[0],
            "Fold 2": scores[1],
            "Fold 3": scores[2],
            "Fold 4": scores[3],
            "Fold 5": scores[4]
        }
    )

cv_df = pd.DataFrame(cv_results)

print("\nCross Validation Results:")
print(cv_df.to_string(index=False))

cv_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cross_validation_results.csv"
    ),
    index=False
)


# ============================================================
# 20. SAVE BEST MODEL PREDICTIONS
# ============================================================

print("\n[16] Saving Predictions")

prediction_df = X_test.copy()

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = best_predictions

prediction_df["Prediction_Probability"] = (
    best_probabilities
)

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "best_model_predictions.csv"
    ),
    index=False
)


# ============================================================
# 21. EDA - SURVIVAL DISTRIBUTION
# ============================================================

print("\n[17] Creating EDA Visualizations")

plt.figure(figsize=(7, 5))

df_clean[target].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Titanic Survival Distribution")

plt.xlabel(
    "Survival Status (0 = No, 1 = Yes)"
)

plt.ylabel("Number of Passengers")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_survival_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 22. SURVIVAL BY GENDER
# ============================================================

if "sex" in df_clean.columns:

    gender_survival = pd.crosstab(
        df_clean["sex"],
        df_clean[target],
        normalize="index"
    )

    gender_survival.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title(
        "Survival Rate by Gender"
    )

    plt.xlabel("Gender")

    plt.ylabel("Survival Rate")

    plt.xticks(
        rotation=0
    )

    plt.legend(
        ["Did Not Survive", "Survived"]
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "02_survival_by_gender.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 23. SURVIVAL BY CLASS
# ============================================================

if "pclass" in df_clean.columns:

    class_survival = pd.crosstab(
        df_clean["pclass"],
        df_clean[target],
        normalize="index"
    )

    class_survival.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title(
        "Survival Rate by Passenger Class"
    )

    plt.xlabel("Passenger Class")

    plt.ylabel("Survival Rate")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "03_survival_by_class.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 24. AGE DISTRIBUTION
# ============================================================

if "age" in df_clean.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df_clean["age"].dropna(),
        bins=30
    )

    plt.title(
        "Age Distribution of Passengers"
    )

    plt.xlabel("Age")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "04_age_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 25. FARE DISTRIBUTION
# ============================================================

if "fare" in df_clean.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df_clean["fare"].dropna(),
        bins=30
    )

    plt.title(
        "Fare Distribution"
    )

    plt.xlabel("Fare")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "05_fare_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 26. GENDER + CLASS ANALYSIS
# ============================================================

if (
    "sex" in df_clean.columns
    and "pclass" in df_clean.columns
):

    grouped = (
        df_clean
        .groupby(["sex", "pclass"])[target]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="pclass",
        columns="sex",
        values=target
    )

    pivot.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title(
        "Survival Rate by Gender and Passenger Class"
    )

    plt.xlabel("Passenger Class")

    plt.ylabel("Survival Rate")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "06_gender_class_survival.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 27. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

print("\n[18] Feature Importance")

rf_model = models["Random Forest"]

rf_preprocessor = rf_model.named_steps[
    "preprocessor"
]

rf_classifier = rf_model.named_steps[
    "model"
]

try:

    feature_names = (
        rf_preprocessor
        .get_feature_names_out()
    )

    importances = (
        rf_classifier.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        "\nTop 15 Important Features:"
    )

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        dpi=300
    )

    plt.close()

except Exception as e:

    print(
        "Feature importance could not be generated:",
        e
    )


# ============================================================
# 28. SAVE CLEANED DATA
# ============================================================

df_clean.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cleaned_titanic_data.csv"
    ),
    index=False
)


# ============================================================
# 29. PROJECT SUMMARY
# ============================================================

best_row = results_df.iloc[0]

summary = f"""
WEEK 6 DATA SCIENCE CAPSTONE PROJECT
Titanic Survival Prediction

Dataset Shape:
{df.shape}

Target Column:
{target}

Training Samples:
{len(X_train)}

Testing Samples:
{len(X_test)}

Best Model:
{best_model_name}

Best Model Performance:

Accuracy  : {best_row['Accuracy']:.4f}
Precision : {best_row['Precision']:.4f}
Recall    : {best_row['Recall']:.4f}
F1 Score  : {best_row['F1']:.4f}
ROC-AUC   : {best_row['ROC-AUC']:.4f}

Important Data Science Note:
The original dataset contains an 'alive' column that directly
represents the survival outcome. This column was excluded from
predictive modelling to prevent target leakage.

The final models therefore use leakage-free predictors.
"""

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(summary)

with open(
    os.path.join(
        OUTPUT_DIR,
        "project_summary.txt"
    ),
    "w"
) as file:

    file.write(summary)


# ============================================================
# 30. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CAPSTONE PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nAll outputs have been saved in: {OUTPUT_DIR}"
)

print("\nGenerated files include:")

for file_name in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(" -", file_name)

print("\nBest Model:", best_model_name)

print(
    "\nThank you - Week 6 Capstone completed."
)