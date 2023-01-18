from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

from classifier import ClassificationModel
from config import (
    params,
    BINARY_CLASSIFIER,
    PRE_PROCESS,
    SAMPLING,
    DF_TRAIN,
    DF_TEST,
    CROSS_VALIDATION,
    RANDOM_SEED,
    TRAIN_SIZE,
    LABEL_COLUMN,
    experiment,
    Classes,
)
from evaluator import BinaryEvaluator, Evaluator, plot_confusion_matrix
from utils import (
    get_tag,
    read_dataset,
    display_averaged_result,
    display_aggregated_result,
    initLog,
)


def binary_runner(df_train, df_test, df_valid, sampling):
    total_true_labels = []
    total_pred_labels = []

    for name in Classes.get_names():
        print(
            f"\nCURRENT MODEL {name} --------------------------------------------------->"
        )

        classification_model = ClassificationModel(sampling, name)
        classification_model.train_model(df_train, df_valid)

        results, current_f1, true_labels, predictions = classification_model.test_model(
            df_test, evaluator=BinaryEvaluator
        )
        total_true_labels.append(true_labels)
        total_pred_labels.append(predictions)

    total_true_labels = np.array(total_true_labels)
    total_pred_labels = np.array(total_pred_labels)

    actual = np.array(
        [total_true_labels[:, i] for i in range(0, total_true_labels.shape[1])]
    )
    pred = np.array(
        [total_pred_labels[:, i] for i in range(0, total_pred_labels.shape[1])]
    )

    accuracy = accuracy_score(actual.argmax(axis=1), pred.argmax(axis=1))
    precision, recall, f1, _ = precision_recall_fscore_support(actual, pred)
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    experiment.log_metrics(results)
    report = classification_report(
        actual, pred, target_names=Classes.get_names(), digits=3, zero_division=0
    )
    experiment.log_text(report)

    print("\n", report)
    plot_confusion_matrix(
        actual.argmax(axis=1), pred.argmax(axis=1), classes=Classes.get_names()
    )

    plot_confusion_matrix(
        actual.argmax(axis=1),
        pred.argmax(axis=1),
        classes=Classes.get_names(),
        normalize=True,
    )

    return results, actual, pred


def cross_validate(train_data, df_test, df_valid, sample_type=None):
    initLog()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    model_results = []
    aggregate_true_labels, aggregate_predictions = [], []

    for i, (train, test) in enumerate(
        skf.split(train_data.review, train_data[LABEL_COLUMN])
    ):
        print(
            "\nFOLD NUMBER -------------------------------------------------> ", i + 1
        )
        df_train = train_data.iloc[train]

        if BINARY_CLASSIFIER:
            results, true_labels, predictions = binary_runner(
                df_train, df_test, df_valid, sample_type
            )
            model_results.append(results)
            aggregate_true_labels.extend(true_labels)
            aggregate_predictions.extend(predictions)

        else:
            classification_model = ClassificationModel(sample_type)
            classification_model.train_model(df_train, df_valid)

            # Test and store best model and results
            (
                results,
                current_f1,
                true_labels,
                predictions,
            ) = classification_model.test_model(df_test, Evaluator)
            model_results.append(results)
            aggregate_true_labels.extend(true_labels)
            aggregate_predictions.extend(predictions)

    target_names = Classes.get_names()
    display_averaged_result(model_results, target_names=target_names)
    display_aggregated_result(
        aggregate_true_labels, aggregate_predictions, target_names
    )

    if BINARY_CLASSIFIER:
        aggregate_true_labels = np.array(aggregate_true_labels).argmax(axis=1)
        aggregate_predictions = np.array(aggregate_predictions).argmax(axis=1)

    plot_confusion_matrix(
        aggregate_true_labels, aggregate_predictions, classes=target_names
    )

    plot_confusion_matrix(
        aggregate_true_labels,
        aggregate_predictions,
        classes=target_names,
        normalize=True,
    )

    experiment.log_confusion_matrix(
        aggregate_true_labels, aggregate_predictions, labels=target_names
    )


def runner(
    df_train,
    df_test,
    binary_classifier=False,
    sampling=None,
    cross_validation=False,
    preprocess=False,
):

    df_test, df_valid = train_test_split(
        df_test,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        stratify=df_test[LABEL_COLUMN],
    )

    if cross_validation:
        cross_validate(df_train, df_test, df_valid, sampling)
        return

    if binary_classifier:
        binary_runner(df_train, df_test, df_valid, sampling)

    else:
        classification_model = ClassificationModel(sampling)
        classification_model.train_model(df_train, df_valid)
        classification_model.test_model(df_test, Evaluator)


if __name__ == "__main__":
    experiment.log_parameters(params)
    experiment.add_tag(get_tag())

    runner(
        df_train=read_dataset(DF_TRAIN),
        df_test=read_dataset(DF_TEST),
        binary_classifier=BINARY_CLASSIFIER,
        sampling=SAMPLING,
        cross_validation=CROSS_VALIDATION,
        preprocess=PRE_PROCESS,
    )
