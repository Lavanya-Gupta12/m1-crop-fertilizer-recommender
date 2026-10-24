"""Stage-2 fertilizer network: model definition, leak-free wrapper, save/load.

Inputs: numeric [temperature, humidity, Moisture, N, K, P] + soil type + fertilizer-dataset crop label.
All preprocessing (StandardScaler, label encoders) is fit inside `fit()` on the training rows only.
"""
import pickle
from pathlib import Path

import numpy as np

NUM_COLS_FERT = ["temperature", "humidity", "Moisture", "N", "K", "P"]
CAT_COLS_FERT = ["soil_type", "label"]
EPOCHS = 40
BATCH = 16


def build_fert_nn(n_num, n_soil, n_crop, n_classes, onehot):
    """Notebook architecture. onehot=False: Embedding(n//2) per categorical (current);
    onehot=True: the categoricals enter as one-hot vectors instead (ablation)."""
    import keras
    from keras import layers
    num_in = layers.Input(shape=(n_num,), name="numeric_input")
    if onehot:
        cat_in = layers.Input(shape=(n_soil + n_crop,), name="onehot_input")
        x = layers.Concatenate()([num_in, cat_in]); inputs = [num_in, cat_in]
    else:
        soil_in = layers.Input(shape=(1,), name="soil_input")
        crop_in = layers.Input(shape=(1,), name="crop_input")
        soil_emb = layers.Flatten()(layers.Embedding(n_soil, max(2, n_soil // 2))(soil_in))
        crop_emb = layers.Flatten()(layers.Embedding(n_crop, max(2, n_crop // 2))(crop_in))
        x = layers.Concatenate()([num_in, soil_emb, crop_emb]); inputs = [num_in, soil_in, crop_in]
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="fert_output")(x)
    model = keras.Model(inputs, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class FertNN:
    """Keras stage-2 model with its preprocessing fitted on the training rows only."""

    def __init__(self, onehot, seed, soil_classes, crop_classes, n_classes):
        from sklearn.preprocessing import LabelEncoder
        self.onehot, self.seed, self.n_classes = onehot, seed, n_classes
        self.soil_enc = LabelEncoder().fit(soil_classes)
        self.crop_enc = LabelEncoder().fit(crop_classes)

    def _inputs(self, df):
        xn = self.scaler.transform(df[NUM_COLS_FERT].values)
        s, c = self.soil_enc.transform(df["soil_type"]), self.crop_enc.transform(df["label"])
        if self.onehot:
            oh = np.hstack([np.eye(len(self.soil_enc.classes_))[s], np.eye(len(self.crop_enc.classes_))[c]])
            return [xn, oh]
        return [xn, s, c]

    def fit(self, df, y):
        import keras
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(df[NUM_COLS_FERT].values)
        keras.utils.set_random_seed(self.seed)
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
        self.model = build_fert_nn(len(NUM_COLS_FERT), len(self.soil_enc.classes_), len(self.crop_enc.classes_),
                                   self.n_classes, self.onehot)
        self.model.fit(self._inputs(df), y, epochs=EPOCHS, batch_size=BATCH, verbose=0)
        return self

    def predict_proba(self, df):
        return np.asarray(self.model.predict(self._inputs(df), verbose=0), dtype=float)

    def predict_proba_fast(self, df):
        """Direct model call: no per-call retracing; for single rows / SHAP batches in the app."""
        import keras
        return np.asarray(keras.ops.convert_to_numpy(self.model(self._inputs(df), training=False)), dtype=float)

    def count_params(self):
        return self.model.count_params()

    # ------------------------------------------------------------ persistence
    def save(self, directory, stem="fert_nn"):
        d = Path(directory)
        self.model.save(d / f"{stem}.keras")
        state = {k: v for k, v in self.__dict__.items() if k != "model"}
        with open(d / f"{stem}_preprocessing.pkl", "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, directory, stem="fert_nn"):
        import keras
        d = Path(directory)
        obj = cls.__new__(cls)
        with open(d / f"{stem}_preprocessing.pkl", "rb") as fh:
            obj.__dict__.update(pickle.load(fh))
        obj.model = keras.models.load_model(d / f"{stem}.keras")
        return obj
