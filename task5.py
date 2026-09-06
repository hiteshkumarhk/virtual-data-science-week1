import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

print("=" * 70)
print("TASK 5 - DEEP LEARNING APPLICATION IN DATA SCIENCE")
print("TITANIC SURVIVAL PREDICTION USING NEURAL NETWORK")
print("=" * 70)

DATA_FILE = "titanic_cleaned_week1.csv"
RESULTS_DIR = "week5_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset '{DATA_FILE}' was not found. "
        "Please keep the CSV file in the same folder as task5.py."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. IDENTIFY TARGET COLUMN
# ============================================================

possible_targets = [
    "Survived",
    "survived",
    "Survival",
    "survival"
]

target = None

for column in possible_targets:
    if column in df.columns:
        target = column
        break

if target is None:
    raise ValueError(
        "Target column 'Survived' was not found. "
        "Please check your dataset."
    )

print("\nTarget variable:", target)


# ============================================================
# 4. REMOVE UNNECESSARY COLUMNS
# ============================================================

drop_columns = []

for column in ["PassengerId", "Name", "Ticket", "Cabin"]:
    if column in df.columns:
        drop_columns.append(column)

X = df.drop(columns=[target] + drop_columns)
y = df[target].copy()

print("\nRemoved columns:", drop_columns)

print("\nFeatures used:")
print(X.columns.tolist())


# ============================================================
# 5. CONVERT TARGET TO NUMERIC
# ============================================================

if y.dtype == "object" or str(y.dtype) == "category":
    y = y.astype("category").cat.codes

y = pd.to_numeric(y)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. DATA PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)


# ============================================================
# 9. TRANSFORM DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

y_train_array = np.asarray(
    y_train,
    dtype=np.float32
)

y_test_array = np.asarray(
    y_test,
    dtype=np.float32
)

print("\nPreprocessing completed!")

print("Processed training shape:",
      X_train_processed.shape)

print("Processed testing shape:",
      X_test_processed.shape)


# ============================================================
# 10. BUILD NEURAL NETWORK
# ============================================================

input_features = X_train_processed.shape[1]

model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(input_features,)
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nNeural Network Architecture:")
model.summary()


# ============================================================
# 12. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING MODEL TRAINING")
print("=" * 70)

history = model.fit(
    X_train_processed,
    y_train_array,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print("\nModel training completed!")


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test_processed,
    y_test_array,
    verbose=0
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))


# ============================================================
# 15. MAKE PREDICTIONS
# ============================================================

prediction_probabilities = model.predict(
    X_test_processed,
    verbose=0
).ravel()

predictions = (
    prediction_probabilities >= 0.5
).astype(int)


# ============================================================
# 16. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_array,
    predictions
)

precision = precision_score(
    y_test_array,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test_array,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_array,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_array,
    prediction_probabilities
)

print("\nPerformance Metrics:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test_array,
    predictions,
    zero_division=0
)

print(report)


# ============================================================
# 18. SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write(
        "TASK 5 - TITANIC SURVIVAL PREDICTION\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        "Model: Neural Network using TensorFlow/Keras\n\n"
    )

    file.write(report)

    file.write("\n\nPerformance Metrics\n")
    file.write("-" * 40 + "\n")

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_array,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Confusion Matrix - Deep Learning Neural Network"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "01_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("\nConfusion matrix saved.")


# ============================================================
# 20. TRAINING VS VALIDATION ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "02_training_validation_accuracy.png"
    ),
    dpi=300
)

plt.close()

print("Accuracy graph saved.")


# ============================================================
# 21. TRAINING VS VALIDATION LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "03_training_validation_loss.png"
    ),
    dpi=300
)

plt.close()

print("Loss graph saved.")


# ============================================================
# 22. SAVE MODEL
# ============================================================

model.save(
    os.path.join(
        RESULTS_DIR,
        "titanic_deep_learning_model.keras"
    )
)

print("Trained model saved.")


# ============================================================
# 23. SAVE PERFORMANCE RESULTS
# ============================================================

results = {
    "Model": "Deep Neural Network",
    "Test Loss": test_loss,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc,
    "Training Epochs": len(history.history["loss"])
}

results_df = pd.DataFrame([results])

results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "deep_learning_results.csv"
    ),
    index=False
)

print("Performance results saved.")


# ============================================================
# 24. SAVE PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["Actual_Survival"] = y_test.values

prediction_output["Predicted_Survival"] = predictions

prediction_output["Survival_Probability"] = (
    prediction_probabilities
)

prediction_output.to_csv(
    os.path.join(
        RESULTS_DIR,
        "deep_learning_predictions.csv"
    ),
    index=False
)

print("Predictions saved.")


# ============================================================
# 25. SAVE MODEL SUMMARY
# ============================================================

with open(
    os.path.join(
        RESULTS_DIR,
        "model_summary.txt"
    ),
    "w"
) as file:

    model.summary(
        print_fn=lambda line: file.write(
            line + "\n"
        )
    )


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TASK 5 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nModel: Deep Neural Network")

print(
    "Test Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print("\nResults saved inside:")
print(RESULTS_DIR)

print("\nGenerated files:")

for file_name in sorted(
    os.listdir(RESULTS_DIR)
):
    print("-", file_name)

print("\n" + "=" * 70)