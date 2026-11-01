# Breast Cancer Classification

An academic machine-learning project comparing Logistic Regression, Support Vector Machine (SVM), and Random Forest on the Wisconsin Breast Cancer Diagnostic dataset. Includes a reproducible training/evaluation script and a Gradio interface for demonstration.

> **Important:** This is an educational prototype, not a medical device. It has not been clinically validated and must not be used for diagnosis, screening, treatment decisions, or medical advice.

## Project overview

Binary classification of diagnostic measurements into **Malignant** (class `0`) and **Benign** (class `1`). The dataset contains 569 samples and 30 numerical features and is loaded from scikit-learn using `load_breast_cancer()`.

## Methodology

1. Load the dataset and its 30 named features.
2. Create a stratified 80/20 train/test split (`random_state=42`).
3. Train Logistic Regression (with `StandardScaler`), Random Forest (200 trees), and SVM (with standardisation).
4. Compare accuracy, malignant-class precision, recall, F1-score, and ROC-AUC.
5. Select the model with the highest malignant-class F1-score.

Malignant is treated as the positive class for class-specific metrics.

## Reproduced results

| Model | Accuracy | Malignant precision | Malignant recall | Malignant F1 | Malignant ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.9825** | **0.9762** | **0.9762** | **0.9762** | **0.9954** |
| SVM | 0.9649 | 0.9318 | 0.9762 | 0.9535 | 0.9940 |
| Random Forest | 0.9474 | 0.9286 | 0.9286 | 0.9286 | 0.9937 |

Logistic Regression achieved **98.25% (approximately 98.2%) accuracy** on this particular 114-sample test split.

### Logistic Regression confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual / Predicted | Malignant | Benign |
|---|---:|---:|
| Malignant | 41 | 1 |
| Benign | 1 | 71 |

### Evaluation caveat

The same hold-out split is used both to compare/select the model and to report metrics. This makes the result exploratory and can introduce model-selection bias. For a stronger final estimate, use cross-validation for model selection and evaluate once on a separate, untouched test set. Benchmark performance does not establish real-world clinical performance.

## Structure

```text
Breast-Cancer-Classification/
├── README.md
├── requirements.txt
├── app.py
├── .gitignore
├── src/
│   ├── __init__.py
│   └── train.py
├── results/
│   ├── model_comparison.csv
│   └── selected_model_classification_report.csv
└── models/  # generated locally; excluded from Git
```

## Run locally

Python 3.10+ recommended.

```bash
cd ML-Fundamentals/Breast-Cancer-Classification
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m src.train
python app.py
```

The training script generates `models/selected_model.joblib` and evaluation outputs in `results/`. The model artifact is excluded from version control; run training before launching the app. Gradio prints the local app URL in the terminal.

## Interactive demo

The Gradio app accepts 30 numerical values as a JSON array in the feature order shown in the interface, or a JSON object keyed by feature names. Returned scores are model outputs, not clinical probabilities.

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · Joblib · Gradio

## Dataset references

- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Image analysis and machine learning applied to breast cancer diagnosis and prognosis*. Analytical and Quantitative Cytology and Histology.
- [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [scikit-learn load_breast_cancer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## Future work

- Use nested cross-validation or a separate untouched test set.
- Add calibration and decision-threshold analysis.
- Add automated tests for preprocessing and prediction input validation.
- Compare repeated stratified splits and report variability.
