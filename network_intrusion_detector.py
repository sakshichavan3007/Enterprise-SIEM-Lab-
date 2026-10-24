#!/usr/bin/env python3
"""
ML Network Intrusion Detector
Train and evaluate a scikit-learn classifier on a labeled network-flow CSV.

Expected input:
- One row per network flow/connection.
- Numeric and/or categorical feature columns.
- A target/label column, e.g. "label", "Label", "class", or "attack_cat".
- The label column should distinguish benign/normal traffic from attack classes,
  or contain multiple attack categories.

This project does not capture live traffic. It analyzes an existing dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

LOG = logging.getLogger("network_intrusion_detector")
COMMON_LABELS = ("label", "Label", "class", "Class", "target", "Target",
                 "attack", "Attack", "attack_cat", "category", "is_attack")


def find_label_column(frame: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(
                f"Label column '{requested}' not found. Available columns: "
                + ", ".join(map(str, frame.columns))
            )
        return requested
    for name in COMMON_LABELS:
        if name in frame.columns:
            return name
    raise ValueError(
        "Could not infer the target column. Pass --label-column NAME. "
        f"Available columns: {', '.join(map(str, frame.columns))}"
    )


def load_dataset(path: Path, label_column: Optional[str], sample_limit: Optional[int]):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError("The dataset contains no rows.")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    target = find_label_column(frame, label_column)
    frame = frame.dropna(subset=[target]).copy()
    if sample_limit and len(frame) > sample_limit:
        frame = frame.sample(n=sample_limit, random_state=42)
    y = frame.pop(target)
    # Avoid training on identifiers that commonly leak dataset-specific identity.
    leak_names = {"id", "flow_id", "src_ip", "source_ip", "dst_ip", "destination_ip",
                  "timestamp", "time_stamp"}
    drop_cols = [c for c in frame.columns if str(c).strip().lower() in leak_names]
    if drop_cols:
        LOG.info("Dropping likely identifier/leakage columns: %s", drop_cols)
        frame = frame.drop(columns=drop_cols)
    # Drop columns that are entirely missing.
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")
    return frame, y, target


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1
    )
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def train(args):
    X, y, label_name = load_dataset(Path(args.data), args.label_column, args.sample_limit)
    # Stratification is ideal for imbalanced intrusion datasets, but can fail when
    # a class has too few samples. Fall back with a clear warning in that case.
    stratify = y if y.value_counts().min() >= 2 else None
    if stratify is None:
        LOG.warning("At least one class has fewer than two rows; using unstratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=stratify
    )
    model = make_pipeline(X_train)
    LOG.info("Training on %d rows and %d features.", len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "dataset": str(Path(args.data)),
        "label_column": label_name,
        "rows_used": int(len(X)),
        "features_used": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classes": [str(v) for v in model.named_steps["classifier"].classes_],
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=model.named_steps["classifier"].classes_).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "network_intrusion_model.joblib"
    metrics_path = output_dir / "evaluation_metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n=== Evaluation Summary ===")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Weighted precision:{metrics['precision_weighted']:.4f}")
    print(f"Weighted recall:   {metrics['recall_weighted']:.4f}")
    print(f"Weighted F1:        {metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Saved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")


def predict(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
    model = joblib.load(model_path)
    data_path = Path(args.data)
    frame = pd.read_csv(data_path, low_memory=False)
    # If an actual label column is present, exclude it from prediction features.
    for label in COMMON_LABELS:
        if label in frame.columns:
            frame = frame.drop(columns=[label])
            break
    frame = frame.replace([np.inf, -np.inf], np.nan)
    predictions = model.predict(frame)
    result = frame.copy()
    result["predicted_class"] = predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)
        result["confidence"] = probabilities.max(axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Wrote {len(result)} predictions to {out}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or run a scikit-learn network intrusion classifier."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="Train and evaluate on a labeled CSV dataset.")
    p_train.add_argument("--data", required=True, help="Path to the labeled CSV dataset.")
    p_train.add_argument("--label-column", default=None, help="Target column name; inferred if omitted.")
    p_train.add_argument("--test-size", type=float, default=0.20, help="Test split fraction (default: 0.20).")
    p_train.add_argument("--sample-limit", type=int, default=None, help="Optional maximum rows for quick experiments.")
    p_train.add_argument("--output-dir", default="artifacts", help="Directory for model and metrics.")
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", help="Predict classes for an unlabeled CSV.")
    p_pred.add_argument("--data", required=True, help="Path to feature CSV.")
    p_pred.add_argument("--model", default="artifacts/network_intrusion_model.joblib", help="Saved model path.")
    p_pred.add_argument("--output", default="artifacts/predictions.csv", help="Prediction CSV output path.")
    p_pred.set_defaults(func=predict)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "test_size", 0.2) <= 0 or getattr(args, "test_size", 0.2) >= 1:
        parser.error("--test-size must be greater than 0 and less than 1.")
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
