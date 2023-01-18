from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)
import numpy as np
import matplotlib.pyplot as plt

from config import experiment, Classes


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
    else:
        title = "Confusion matrix, without normalization"

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("True label", fontsize=15)
    ax.set_ylabel("Predicted label", fontsize=15)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.show()
    return ax


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_classifier(self, split_train_dataset):
        test_texts, true_labels = split_train_dataset
        predictions = self.model.predict(test_texts)
        predictions = [np.argmax(i) for i in predictions]

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions
        )

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        experiment.log_metrics(results)
        report = classification_report(
            true_labels,
            predictions,
            target_names=Classes.get_names(),
            digits=3,
            zero_division=0,
        )
        experiment.log_text(report)
        print("\n", report)

        current_f1 = max(results.get("f1"))
        plot_confusion_matrix(true_labels, predictions, classes=Classes.get_names())

        return results, current_f1, true_labels, predictions


class BinaryEvaluator(Evaluator):
    def evaluate_classifier(self, split_train_dataset):
        test_texts, true_labels = split_train_dataset
        predictions = self.model.predict(test_texts)
        predictions = [np.argmax(i) for i in predictions]

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions
        )

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        experiment.log_metrics(results)
        report = classification_report(true_labels, predictions)
        experiment.log_text(report)
        print("\n", report)

        current_f1 = results.get("f1")[1]
        return results, current_f1, true_labels, predictions
