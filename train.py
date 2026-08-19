import os
import joblib
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. Load Dataset
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Target classes:", target_names)


# ============================================================
# 2. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Feature Preprocessing
# ============================================================

numeric_features = list(range(X.shape[1]))

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 4. Random Forest Model
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. Create ML Pipeline
# ============================================================

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)


# ============================================================
# 6. Train Pipeline
# ============================================================

print("\nTraining model...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 7. Evaluate
# ============================================================

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)


# ============================================================
# 8. Save Model
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = (
    "models/breast_cancer_rf_pipeline.joblib"
)

joblib.dump(
    model_pipeline,
    model_path
)


print("\nModel saved successfully:")
print(model_path)


# ============================================================
# 9. Save Metadata
# ============================================================

metadata = {
    "model_type": "RandomForestClassifier",
    "dataset": "Breast Cancer Wisconsin",
    "features": feature_names.tolist(),
    "target_names": target_names.tolist(),
    "n_features": X.shape[1],
    "model_version": "1.0.0"
}

joblib.dump(
    metadata,
    "models/model_metadata.joblib"
)

print("Metadata saved.")