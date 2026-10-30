"""Very first ML project: classify iris flower species from measurements.

Uses a bundled teaching dataset (no network). Prints held-out accuracy and
confusion matrix. Not research or a competition result.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def run(output_dir: Path | None = None) -> dict[str, float]:
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    # Scaling happens *inside* the pipeline: the test set is never used to fit it.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    baseline = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)

    pred = model.predict(X_test)
    result = {
        "holdout_accuracy": float(accuracy_score(y_test, pred)),
        "majority_baseline_accuracy": float(accuracy_score(y_test, baseline.predict(X_test))),
        "training_cv_accuracy": float(cross_val_score(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
        ).mean()),
    }
    print("Dataset: Iris (150 flowers, four measured features, three species)")
    print(f"Train examples: {len(y_train)}; test examples: {len(y_test)}")
    print("Accuracy:", result)
    print("\nHeld-out test classification report:\n", classification_report(y_test, pred, target_names=data.target_names))
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=data.target_names)
        plt.title("Iris: held-out test confusion matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
    return result


if __name__ == "__main__":
    run(output_dir=Path("output"))
