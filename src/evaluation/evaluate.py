
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained classification pipeline on test data."""

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="macro"
        ),
        "Recall": recall_score(
            y_test, y_pred, average="macro"
        ),
        "F1-score": f1_score(
            y_test, y_pred, average="macro"
        )
    }

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    labels = model.classes_

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    return y_pred, metrics, report, confusion_df