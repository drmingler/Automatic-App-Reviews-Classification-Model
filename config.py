from datetime import datetime
from enum import Enum

from comet_ml import Experiment
from transformers import (
    TFBertModel,
    BertTokenizer,
    RobertaTokenizer,
    TFRobertaModel,
    TFDistilBertModel,
    DistilBertTokenizer,
    TFXLNetModel,
    XLNetTokenizer,
)

# Constants
TEXT_COLUMN = "review"
LABEL_COLUMN = "task"
DEFAULT_MODEL_NAME = "Multiclass_Model"

# Model Hyper Parameters
# PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
# TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# PRE_TRAINED_MODEL_NAME = "roberta-base"
# TOKENIZER = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# PRE_TRAINED_MODEL_NAME = "distilbert-base-uncased"
# TOKENIZER = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

PRE_TRAINED_MODEL_NAME = "xlnet-base-cased"
TOKENIZER = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

MAX_LEN = 48
RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 16
TRAIN_SIZE = 0.7

# Model Runner Configs
PROJECT_NAME = "XLNET EXPERIMENTS WITH AUGMENTATION"
UNIQUE_IDENTIFIER = "DATASPLIT_GUZMAN"
CROSS_VALIDATION = False
BINARY_CLASSIFIER = False
SAMPLING = None  # undersample, oversample or None
CNN = True
PRE_PROCESS = False

# File Path definitions
PATH = "PATH TO DATASET"
MODEL_SAVE_PATH = f"{PATH}FINAL_DATASETS/"
MODEL_LOG_PATH = f"{PATH}FINAL_AUG_LOG/"
LOG_RESULT_TIME = datetime.now().strftime("%Y%m%d-%H%M")

# Training and testing dataset
DF_TRAIN = f"{MODEL_SAVE_PATH}augmented.csv"
DF_TEST = f"{MODEL_SAVE_PATH}dataset_guzman_relabelled.csv"

params = {
    "TEXT_COLUMN": TEXT_COLUMN,
    "LABEL_COLUMN": LABEL_COLUMN,
    "DEFAULT_MODEL_NAME": DEFAULT_MODEL_NAME,
    "CNN": CNN,
    "MAX_LEN": MAX_LEN,
    "RANDOM_SEED": RANDOM_SEED,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "TRAIN_SIZE": TRAIN_SIZE,
    "BINARY_CLASSIFIER": BINARY_CLASSIFIER,
    "CROSS_VALIDATION": CROSS_VALIDATION,
    "PRE_PROCESS": PRE_PROCESS,
    "PRE_TRAINED_MODEL_NAME": PRE_TRAINED_MODEL_NAME,
    "SAMPLING": SAMPLING,
    "TRAIN_PATH": DF_TRAIN,
    "TEST_PATH": DF_TEST or "",
}

# Comet Logger
experiment = Experiment(
    api_key="LOGGER API KEY",
    project_name=PROJECT_NAME,
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
)


class Classes(Enum):
    PD = 0
    RT = 1
    FR = 2
    UE = 3

    @classmethod
    def get_names(cls):
        return [i.name for i in cls]
