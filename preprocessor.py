from config import LABEL_COLUMN, RANDOM_SEED, TOKENIZER, TEXT_COLUMN, MAX_LEN, Classes
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np

def convert_labels_to_binary(dataset, label):
    dataset[LABEL_COLUMN] = dataset[LABEL_COLUMN].apply(
        lambda x: 1 if x == label else 0
    )
    return dataset


def convert_labels_to_multi_class(dataset):
    label_names = Classes.get_names()
    dataset[LABEL_COLUMN] = dataset[LABEL_COLUMN].apply(lambda x: label_names.index(x))
    return dataset


def encode_dataset(tokenizer, review, max_len):
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="np",
    )

    return encoding.input_ids, encoding.attention_mask


def undersample(df):
    classes = df[LABEL_COLUMN].value_counts().to_dict()
    least_class_amount = min(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df[LABEL_COLUMN] == key])
    classes_sample = []
    for i in range(0, len(classes_list) - 1):
        classes_sample.append(
            classes_list[i].sample(least_class_amount, random_state=RANDOM_SEED)
        )
    df_maybe = pd.concat(classes_sample, ignore_index=True)
    final_df = pd.concat([df_maybe, classes_list[-1]], axis=0, ignore_index=True)
    final_df = final_df.reset_index(drop=True)
    return final_df


def oversample(df):
    classes = df[LABEL_COLUMN].value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df[LABEL_COLUMN] == key])
    classes_sample = []
    for i in range(1, len(classes_list)):
        classes_sample.append(
            classes_list[i].sample(most, replace=True, random_state=RANDOM_SEED)
        )
    df_maybe = pd.concat(classes_sample, ignore_index=True)
    final_df = pd.concat([df_maybe, classes_list[0]], axis=0, ignore_index=True)
    final_df = final_df.reset_index(drop=True)
    return final_df


def preprocess(tokenizer, reviews, max_len):
    input_ids = []
    attention_masks = []
    for review in reviews:
        input_id, attention_mask = encode_dataset(tokenizer, review, max_len)
        input_ids.append(input_id.flatten())
        attention_masks.append(attention_mask.flatten())

    return np.array(input_ids), np.array(attention_masks)


class Preprocess:
    def __init__(self, sample_type=None, label=None):
        self.sample_type = sample_type
        self.label = label

    def process_training_data(self, df_train, df_valid):
        X_train = preprocess(TOKENIZER, df_train[TEXT_COLUMN], MAX_LEN)
        Y_train = to_categorical(df_train[LABEL_COLUMN].to_numpy())

        X_valid = preprocess(TOKENIZER, df_valid[TEXT_COLUMN], MAX_LEN)
        Y_valid = to_categorical(df_valid[LABEL_COLUMN].to_numpy())

        return X_train, Y_train, X_valid, Y_valid

    def prepare_training_data(self, train_data, valid_data):
        train_data = train_data.copy()
        valid_data = valid_data.copy()
        df_train = self.convert_labels(train_data, self.label)
        df_valid = self.convert_labels(valid_data, self.label)

        print("BEFORE SAMPLING ", df_train[LABEL_COLUMN].value_counts())
        if self.sample_type and self.sample_type == "undersample":
            df_train = undersample(df_train)

        if self.sample_type and self.sample_type == "oversample":
            df_train = oversample(df_train)

        print("AFTER SAMPLING ", df_train[LABEL_COLUMN].value_counts())

        processed_data = self.process_training_data(df_train, df_valid)
        return processed_data

    def prepare_testing_data(self, test_dataset):
        test_data = test_dataset.copy()
        test_data = self.convert_labels(test_data, self.label)
        test_texts = preprocess(TOKENIZER, test_data[TEXT_COLUMN], MAX_LEN)
        test_labels = test_data[LABEL_COLUMN].to_numpy()
        return test_texts, test_labels

    def convert_labels(self, data, label):
        return convert_labels_to_multi_class(data)


class BinaryPreprocess(Preprocess):
    def convert_labels(self, data, label):
        return convert_labels_to_binary(data, label)
