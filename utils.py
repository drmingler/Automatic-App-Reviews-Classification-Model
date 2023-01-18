import os
from config import *
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)


# Create and Log Tag
def get_tag():
    classifier = "Multi-label" if BINARY_CLASSIFIER else "Multi-class"
    architecture = "CNN" if CNN else "FNN"
    pre_process = "preprocessed" if PRE_PROCESS else "not_preprocessed"
    sampling = SAMPLING or "NoSampling"

    return UNIQUE_IDENTIFIER + "_" + classifier + "_" + sampling + "_" + architecture


# Log some texts
def initLog():
    print(MODEL_LOG_PATH)

    if not os.path.isdir(MODEL_LOG_PATH):
        print("Log folder does not exist, trying to create folder.")
        try:
            os.mkdir(MODEL_LOG_PATH)
        except OSError:
            print("Creation of the directory %s failed" % MODEL_LOG_PATH)
        else:
            print("Successfully created the directory %s" % MODEL_LOG_PATH)

    logfile = MODEL_LOG_PATH + get_log_file()
    log_txt = datetime.now().strftime("%Y-%m-%d %H:%M") + " " + get_info()
    with open(logfile, "a") as log:
        log.write(log_txt + "\n")


def logResult(result):
    logfile = MODEL_LOG_PATH + get_log_file()
    with open(logfile, "a") as log:
        log.write(result + "\n")


def get_info():
    model_config = (
        "model: {}, max_len: {}, epochs: {}, batch_size: {}, train_size: {}, Seed: {}, train_path : {} "
        "test_path : {}".format(
            PRE_TRAINED_MODEL_NAME,
            MAX_LEN,
            EPOCHS,
            BATCH_SIZE,
            TRAIN_SIZE,
            RANDOM_SEED,
            DF_TRAIN,
            DF_TEST,
        )
    )
    return model_config


def get_log_file():
    classifier = "binary" if BINARY_CLASSIFIER else "multiclass"
    cross_val = "cross_validation" if CROSS_VALIDATION else ""
    pre_process = "preprocessed" if PRE_PROCESS else "not_preprocessed"
    sampling = SAMPLING or "NoSampling"

    return (
        classifier
        + "_"
        + sampling
        + "_"
        + cross_val
        + "_"
        + pre_process
        + "_"
        + LOG_RESULT_TIME
        + ".txt"
    )


def read_dataset(path):
    return pd.read_csv(path)


def save_model(model, model_name=None):
    model_name = model_name or DEFAULT_MODEL_NAME
    model_file_path = f"{PATH}saved-models/model-{model_name}.h5"

    # serialize weights to HDF5
    model.save_weights(model_file_path)
    return model_file_path


def load_model(model, model_name=None):
    model_name = model_name or DEFAULT_MODEL_NAME
    model_file_path = f"{PATH}saved-models/model-{model_name}.h5"

    # load weights to HDF5
    model.load_weights(model_file_path)
    return model


def calculate_result(result):
    precision, recall, f1, accuracy = [], [], [], []
    for i in result:
        precision.append(i["precision"])
        recall.append(i["recall"])
        f1.append(i["f1"])
        accuracy.append(i["accuracy"])

    precision = np.array(precision).mean(axis=0)
    recall = np.array(recall).mean(axis=0)
    f1 = np.array(f1).mean(axis=0)
    accuracy = np.array(accuracy).mean()

    return precision, recall, f1, accuracy


def display_averaged_result(result, target_names):
    precision, recall, f1, accuracy = calculate_result(result)

    heading = (
        "------------------------> Averaged Metrics Result <---------------------------"
    )
    print(heading)
    logResult(heading)

    heading = "              precision    recall  f1-score"
    print(heading)
    logResult(heading)

    for i in range(len(target_names)):
        result = (
            "{:<14}".format(target_names[i])
            + "  {:.5f}".format(precision[i])
            + "   {:.5f}".format(recall[i])
            + "   {:.5f}".format(f1[i])
        )
        print(result)
        logResult(result)

    accuracy = "\nTotal Accuracy  {:.5f}".format(accuracy)
    print(accuracy)
    logResult(accuracy)


def display_aggregated_result(
    aggregate_true_labels, aggregate_predictions, target_names
):
    heading = "------------------------> Aggregated Metrics Result <---------------------------"
    print(heading)
    logResult(heading)

    result = classification_report(
        aggregate_true_labels,
        aggregate_predictions,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )
    print(result)
    logResult(result)
