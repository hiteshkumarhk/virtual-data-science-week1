import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# ============================================================
# TASK 4 - SUPERVISED LEARNING: TITANIC SURVIVAL PREDICTION
# ============================================================

print("=" * 70)
print("TASK 4 - TITANIC SURVIVAL PREDICTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

DATA_FILE = "titanic_cleaned_week1.csv"
RESULTS_DIR = "week4_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. Identify Target Column
# ------------------------------------------------------------

possible_targets = ["Survived", "survived", "Survival", "survival"]

target = None

for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    raise ValueError(
        "Target column 'Survived' was not found. "
        "Please check the dataset columns."
    )

print("\nTarget variable:", target)

# ------------------------------------------------------------
# 3. Remove Unnecessary Columns
# ------------------------------------------------------------

drop_columns = []

for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=[target] + drop_columns)
y = df[target]

# Convert target to numeric if necessary
if y.dtype == "object":
    y = y.astype("category").cat.codes

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# ------------------------------------------------------------
# 4. Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 5. Identify Numeric and Categorical Features
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ------------------------------------------------------------
# 6. Preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ------------------------------------------------------------
# 7. Model 1 - Logistic Regression
# ------------------------------------------------------------

logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred, zero_division=0)
logistic_recall = recall_score(y_test, logistic_pred, zero_division=0)
logistic_f1 = f1_score(y_test, logistic_pred, zero_division=0)
logistic_auc = roc_auc_score(y_test, logistic_prob)

# ------------------------------------------------------------
# 8. Model 2 - Random Forest
# ------------------------------------------------------------

random_forest_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=8,
        min_samples_split=5
    ))
])

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_auc = roc_auc_score(y_test, rf_prob)

# ------------------------------------------------------------
# 9. Model Comparison
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],
    "Precision": [
        logistic_precision,
        rf_precision
    ],
    "Recall": [
        logistic_recall,
        rf_recall
    ],
    "F1 Score": [
        logistic_f1,
        rf_f1
    ],
    "ROC-AUC": [
        logistic_auc,
        rf_auc
    ]
})

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    os.path.join(RESULTS_DIR, "model_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 10. Cross-Validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_cv = cross_val_score(
    logistic_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

rf_cv = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n5-Fold Cross-Validation")
print("Logistic Regression:", round(logistic_cv.mean(), 4))
print("Random Forest:", round(rf_cv.mean(), 4))

# ------------------------------------------------------------
# 11. Confusion Matrix - Best Model
# ------------------------------------------------------------

if rf_f1 >= logistic_f1:
    best_model_name = "Random Forest"
    best_predictions = rf_pred
else:
    best_model_name = "Logistic Regression"
    best_predictions = logistic_pred

print("\nBest Model:", best_model_name)

cm = confusion_matrix(y_test, best_predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "01_confusion_matrix.png"),
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 12. Classification Report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(classification_report(y_test, best_predictions))

with open(
    os.path.join(RESULTS_DIR, "classification_report.txt"),
    "w"
) as f:

    f.write("TASK 4 - TITANIC SURVIVAL PREDICTION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Best Model: {best_model_name}\n\n")

    f.write(
        classification_report(
            y_test,
            best_predictions
        )
    )

# ------------------------------------------------------------
# 13. Save Final Results
# ------------------------------------------------------------

results = {
    "Best Model": best_model_name,
    "Logistic Regression Accuracy": logistic_accuracy,
    "Logistic Regression F1": logistic_f1,
    "Logistic Regression ROC-AUC": logistic_auc,
    "Random Forest Accuracy": rf_accuracy,
    "Random Forest F1": rf_f1,
    "Random Forest ROC-AUC": rf_auc,
    "Logistic 5-Fold CV Accuracy": logistic_cv.mean(),
    "Random Forest 5-Fold CV Accuracy": rf_cv.mean()
}

results_df = pd.DataFrame([results])

results_df.to_csv(
    os.path.join(RESULTS_DIR, "final_results.csv"),
    index=False
)

# ------------------------------------------------------------
# 14. Save Predictions
# ------------------------------------------------------------

prediction_output = X_test.copy()
prediction_output["Actual_Survival"] = y_test.values
prediction_output["Predicted_Survival"] = best_predictions

prediction_output.to_csv(
    os.path.join(RESULTS_DIR, "test_predictions.csv"),
    index=False
)

# ------------------------------------------------------------
# 15. Final Output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TASK 4 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nBest Model:", best_model_name)
print("Best Model Accuracy:",
      round(
          max(logistic_accuracy, rf_accuracy),
          4
      ))

print("\nResults saved inside:")
print(RESULTS_DIR)

print("\nGenerated files:")
for file in os.listdir(RESULTS_DIR):
    print("-", file)

print("\n" + "=" * 70)