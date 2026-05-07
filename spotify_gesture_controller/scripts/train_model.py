#!/usr/bin/env python3
"""Train gesture classifiers and save the best model for real-time use."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import GESTURES, MODEL_DIR, PROCESSED_DATA_DIR, RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    X = pd.read_csv(args.processed_dir / "features.csv")
    y_labels = pd.read_csv(args.processed_dir / "labels.csv")["label"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True)),
        ]
    )
    svm_grid = {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.01, 0.1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(svm, svm_grid, cv=cv, scoring="accuracy", n_jobs=1, refit=True)
    search.fit(X_train, y_train)

    models = {
        "svm": search.best_estimator_,
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train),
        "knn": Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]
        ).fit(X_train, y_train),
    }
    scores = {name: accuracy_score(y_test, model.predict(X_test)) for name, model in models.items()}
    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    y_pred = best_model.predict(X_test)
    class_names = list(encoder.classes_)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_dir / "gesture_model.pkl")
    joblib.dump(encoder, args.model_dir / "label_encoder.pkl")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    (args.results_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35, values_format="d"
    )
    ax.set_title(f"Gesture Confusion Matrix ({best_name}, acc={scores[best_name]:.3f})")
    fig.tight_layout()
    fig.savefig(args.results_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    metadata = {
        "best_model": best_name,
        "svm_best_params": search.best_params_,
        "svm_best_cv_accuracy": float(search.best_score_),
        "test_accuracy_by_model": {k: float(v) for k, v in scores.items()},
        "classes": class_names,
        "requested_gestures": GESTURES,
        "feature_columns": list(X.columns),
        "training_window_samples": int(round(float(X["row_count"].median()))) if "row_count" in X else None,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }
    (args.model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.results_dir / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"SVM best parameters: {search.best_params_}")
    print(f"Model scores: {scores}")
    print(f"Saved best model ({best_name}) to {args.model_dir / 'gesture_model.pkl'}")


if __name__ == "__main__":
    main()
