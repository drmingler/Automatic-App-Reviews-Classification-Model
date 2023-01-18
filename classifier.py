from config import (
    BINARY_CLASSIFIER,
    CNN,
    EPOCHS,
    BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    PRE_TRAINED_MODEL_NAME,
    MAX_LEN,
)
import tensorflow as tf

from preprocessor import Preprocess, BinaryPreprocess
from utils import save_model, load_model
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


def get_architecture(outputs, max_length, output_layer):
    if CNN:
        outputs = outputs[0]
        net = tf.keras.layers.Conv1D(max_length, (2), activation="relu")(outputs)
        net = tf.keras.layers.MaxPooling1D(2)(net)

        net = tf.keras.layers.Conv1D(64, (2), activation="relu")(net)
        net = tf.keras.layers.MaxPooling1D(2)(net)

        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(max_length, activation="relu")(net)

        net = tf.keras.layers.Dropout(0.2)(net)
        out = tf.keras.layers.Dense(*output_layer)(net)
        return out

    else:
        # outputs = outputs[1]

        # For distilbert and XLNET get CLS token
        last_hidden_state = outputs[0]
        outputs = last_hidden_state[:, 0, :]

        lay = tf.keras.layers.Dense(max_length, activation="relu")(outputs)
        lay = tf.keras.layers.Dropout(0.2)(lay)
        out = tf.keras.layers.Dense(*output_layer)(lay)
        return out


class Classifier:
    def __init__(self, bert_layer, max_len=512):
        self.bert_layer = bert_layer
        self.max_len = max_len

    def build_model(self):
        input_ids = tf.keras.Input(
            shape=(self.max_len,), dtype=tf.int32, name="input_id"
        )
        attention_masks = tf.keras.Input(
            shape=(self.max_len,), dtype=tf.int32, name="attention_mask"
        )
        outputs = self.bert_layer(input_ids=input_ids, attention_mask=attention_masks)

        if BINARY_CLASSIFIER:
            out = get_architecture(
                outputs, max_length=self.max_len, output_layer=(2, "softmax")
            )
            model = tf.keras.models.Model(
                inputs=[input_ids, attention_masks], outputs=out
            )

            METRICS = [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                loss="binary_crossentropy",
                metrics=METRICS,
            )

        else:
            out = get_architecture(
                outputs, max_length=self.max_len, output_layer=(4, "softmax")
            )
            model = tf.keras.models.Model(
                inputs=[input_ids, attention_masks], outputs=out
            )

            METRICS = [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                loss="categorical_crossentropy",
                metrics=METRICS,
            )

        return model

    def create_and_train_classifier(self, split_train_dataset, review_label=None):
        model = self.build_model()
        X_train, Y_train, X_valid, Y_valid = split_train_dataset
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=3, restore_best_weights=True
        )

        model.fit(
            x=X_train,
            y=Y_train,
            validation_data=(X_valid, Y_valid),
            epochs=EPOCHS,
            callbacks=[early_stopping],
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        self._save_model(model, review_label)
        return model

    @staticmethod
    def _save_model(model, review_label):
        if not review_label:
            model_name = DEFAULT_MODEL_NAME
        else:
            model_name = review_label
        save_model(model, model_name)


class ClassificationModel:
    def __init__(self, sample=None, review_label=None):
        # self.BERT = TFBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        # self.BERT = TFRobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        # self.BERT = TFDistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.BERT = TFXLNetModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.sample = sample
        self.review_label = review_label
        self.model_architecture = Classifier(bert_layer=self.BERT, max_len=MAX_LEN)
        self.model = self.model_architecture.build_model()

    @property
    def pre_processor(self):
        pre_processor = Preprocess(self.sample)
        if self.review_label:
            pre_processor = BinaryPreprocess(self.sample, self.review_label)
        return pre_processor

    def train_model(self, df_train, df_valid):
        split_train_dataset = self.pre_processor.prepare_training_data(
            df_train, df_valid
        )
        self.model = self.model_architecture.create_and_train_classifier(
            split_train_dataset, review_label=self.review_label
        )

    def test_model(self, df_test, evaluator, load_from_memory=False):

        if load_from_memory:
            model = self.model_architecture.build_model()
            self.model = load_model(model_name=self.review_label, model=model)

        split_test_dataset = self.pre_processor.prepare_testing_data(df_test)
        _evaluator = evaluator(self.model)
        return _evaluator.evaluate_classifier(split_test_dataset)
