# ML Network Intrusion Detector

A scikit-learn project that trains a machine-learning classifier to identify network traffic classes (for example, benign traffic versus attacks) from a **labeled CSV dataset**. It uses a preprocessing pipeline for numeric and categorical features, a class-weighted Random Forest classifier, and evaluation metrics.

> **Scope:** This is an offline dataset-analysis project. It does not sniff packets, block traffic, or provide a production intrusion-prevention system.

## Features

- Detects a target/label column automatically or accepts one explicitly.
- Handles numeric and categorical columns, missing values, and unknown categories.
- Trains a Random Forest classifier with `class_weight="balanced"`.
- Reports accuracy, weighted precision, weighted recall, weighted F1, and a classification report.
- Saves a reusable model and evaluation metrics.
- Predicts classes for a separate, unlabeled CSV.

## Repository layout

```text
ml_network_intrusion_detector/
├── src/
│   └── network_intrusion_detector.py
├── docs/
│   ├── requirements.pdf
│   ├── project_report.pdf
│   └── sandip_university_logo.png
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10–3.12 recommended
- Dependencies listed in `requirements.txt`
- A labeled CSV file from a public network intrusion dataset

Examples of public datasets to investigate include **CIC-IDS2017**, **CSE-CIC-IDS2018**, **UNSW-NB15**, and **NSL-KDD**. Check each dataset's license, terms, schema, and appropriate citation before use. Dataset columns and label conventions differ; choose the correct label column for your file.

## Setup

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

Use the CSV path and target column for your dataset:

```bash
python src/network_intrusion_detector.py train \
  --data data/network_flows.csv \
  --label-column Label
```

If the target column has a common name such as `label`, `Label`, `class`, `target`, `attack`, or `attack_cat`, the program attempts to infer it:

```bash
python src/network_intrusion_detector.py train --data data/network_flows.csv
```

Optional arguments:

```text
--test-size 0.20       Fraction reserved for testing (default 20%)
--sample-limit 50000   Use a random subset for faster experiments
--output-dir artifacts Directory for model and metrics
```

Training creates:

- `artifacts/network_intrusion_model.joblib` — trained preprocessing + classifier pipeline
- `artifacts/evaluation_metrics.json` — evaluation metrics and class-wise report

## Predict on new data

The new CSV must contain the same feature columns used during training, with compatible meanings and types. It should normally omit the target column.

```bash
python src/network_intrusion_detector.py predict \
  --data data/new_flows.csv \
  --model artifacts/network_intrusion_model.joblib \
  --output artifacts/predictions.csv
```

The output CSV contains the input features, `predicted_class`, and (when available) `confidence`, the maximum predicted class probability. Confidence is not a guarantee that a prediction is correct.

## Methodology

1. Load the labeled CSV and remove rows without a target label.
2. Remove common identifier/leakage columns such as flow IDs, IP addresses, and timestamps.
3. Impute missing numeric values with the median and categorical values with the most frequent value.
4. One-hot encode categorical features and scale numeric features.
5. Split data into training and testing subsets, stratifying by class when possible.
6. Fit a class-weighted Random Forest and evaluate it on the held-out test set.

## Important evaluation notes

- Network datasets are often highly imbalanced. Accuracy alone can be misleading; inspect per-class precision, recall, F1, and the confusion matrix.
- Avoid leakage: do not include identifiers, future information, or duplicate flows that make test performance unrealistically optimistic.
- A random split may not represent deployment conditions. For stronger evaluation, consider time-based or host-based splits and cross-dataset testing.
- Treat results as a research/educational baseline, not as proof of operational security.
- Review false negatives and false positives with a security specialist before any real-world use.

## Troubleshooting

- **Target column not found:** specify `--label-column YOUR_COLUMN`.
- **Memory use is high:** try `--sample-limit 50000`, reduce feature cardinality, or use a smaller dataset.
- **Prediction feature mismatch:** ensure the inference CSV has the same feature names and compatible columns as the training data.
- **Only one class present:** check the selected target column and dataset filtering.

## Reproducibility and limitations

The split seed and classifier seed are fixed at `42`. Results depend on the dataset, selected features, label quality, and preprocessing. No performance numbers are claimed in this repository; run the pipeline on your chosen dataset and report the generated metrics.

## Author / Institution

Prepared as an educational machine-learning project. The included report uses the supplied Sandip University logo. Add your name, department, course, academic year, dataset citation, and experiment results before formal submission.

## License

No license is assigned. Add a license file if you intend to distribute or reuse this project.
