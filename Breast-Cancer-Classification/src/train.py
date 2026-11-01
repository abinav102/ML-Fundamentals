"""Train and evaluate breast-cancer classification models.

Dataset: scikit-learn's copy of the Wisconsin Breast Cancer Diagnostic dataset.
Target encoding: 0 = malignant, 1 = benign.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Load the 30-feature diagnostic dataset bundled with scikit-learn."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y, list(dataset.target_names)


def build_models():
    """Return the three classifiers used in the experiment."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=RANDOM_STATE,
                class_weight="balanced"),
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    """Calculate metrics with malignant (class 0) as the positive class."""
    predictions = model.predict(X_test)
    class_order = list(model.classes_)
    malignant_probabilities = model.predict_proba(X_test)[
        :, class_order.index(0)
    ]
    malignant_truth = (y_test == 0).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Malignant precision": precision_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "Malignant recall": recall_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "Malignant F1": f1_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "Malignant ROC-AUC": roc_auc_score(
            malignant_truth, malignant_probabilities
        ),
    }


def run_experiment():
    """Train models, save comparison metrics, and persist the selected model."""
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

 trained_models, rows = {}, []
    for ame, model in build_models().items():
 model.fit(X_train, y_train)
 trained_models[ ame] = model
 rows.append(evaluate_model( ame, model, X_test, y_test))

 results = pd.DataFrame(rows).sort_values("Malignant F1", ascending=False)
 results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
 selected_name = results.iloc[0]["Model"]
 selected_model = trained_models[selected_name]
 predictions = selected_model.predict(X_test)

 joblib.dump({
        "model": selected_model,
        "model_name": selected_name,
        "feature_names": list(X.columns),
        "target_names": target_names,
    }, MODELS_DIR / "selected_model.joblib")

 report = classification_report(
 y_test, predictions, labels=[0, 1],
 target_names=["malignant", "benign"],
 zero_division=0, output_dict=True,
    )
 pd.DataFrame(report).transpose().to_csv(
 RESULTS_DIR / "selected_model_classification_report.csv"
    )

 plot = ConfusionMatrixDisplay.from_predictions(
 y_test, predictions, labels=[0, 1],
 display_labels=["Malignant", "Benign"], cmap="Blues",
    )
 plot.ax_.set_title(f"{selected_name} - test split")
 plot.figure_.tight_layout()
 plot.figure_.savefig(
 RESULTS_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight"
    )
 plt.close(plot.figure_)

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Train/test split: {len(X_train)}/{len(X_test)}")
    print(f"Selected by malignant-class F1: {selected_name}")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCaution: the same hold-out split is used to compare/select models.")
    print("Use nested cross-validation or a separate untouched test set for")
    print("a less biased final performance estimate.")
    return results


if __name__ == "__main__":
    run_experiment()
