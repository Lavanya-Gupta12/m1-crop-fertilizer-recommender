"""Re-train the stage-2 fertilizer model exactly as in the notebook (cells [3]-[10]) and save it
to models/ so the app can load it instead of retraining.

This deliberately keeps the notebook's protocol, including its known issues (scaler fit on all
99 rows, 20-row test set, no k-fold); those are fixed in Tasks 2 and 4, not here.

Usage: python scripts/train_stage2_notebook.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

SEED = 42
NUM_COLS_FERT = ["temperature", "humidity", "Moisture", "N", "K", "P"]


def load_fert_df():
    local = ROOT / "data" / "Fertilizer Prediction.csv"
    if local.exists():
        path = local
    else:
        import kagglehub
        path = Path(kagglehub.dataset_download("gdabhishek/fertilizer-prediction")) / "Fertilizer Prediction.csv"
    df = pd.read_csv(path).dropna().drop_duplicates()
    df = df.rename(columns={"Temparature": "temperature", "Humidity ": "humidity", "Soil Type": "soil_type",
                            "Crop Type": "label", "Nitrogen": "N", "Potassium": "K", "Phosphorous": "P",
                            "Fertilizer Name": "fertilizer"})
    df["label"] = df["label"].map(lambda s: str(s).strip().lower().replace(" ", "")).replace({"paddy": "rice"})
    return df


def main():
    keras.utils.set_random_seed(SEED)
    df = load_fert_df()
    fert_crop_encoder = LabelEncoder().fit(sorted(df["label"].unique()))
    soil_encoder = LabelEncoder().fit(sorted(df["soil_type"].unique()))
    fert_encoder = LabelEncoder().fit(sorted(df["fertilizer"].unique()))
    scaler_fert = StandardScaler().fit(df[NUM_COLS_FERT].values)   # notebook protocol (leaky)
    Xn = scaler_fert.transform(df[NUM_COLS_FERT].values)
    Xs = soil_encoder.transform(df["soil_type"])
    Xc = fert_crop_encoder.transform(df["label"])
    y = fert_encoder.transform(df["fertilizer"])
    Xn_tr, Xn_te, Xs_tr, Xs_te, Xc_tr, Xc_te, y_tr, y_te = train_test_split(
        Xn, Xs, Xc, y, test_size=0.2, stratify=y, random_state=SEED)

    n_soil, n_crop = len(soil_encoder.classes_), len(fert_crop_encoder.classes_)
    num_in = layers.Input(shape=(len(NUM_COLS_FERT),), name="numeric_input")
    soil_in = layers.Input(shape=(1,), name="soil_input")
    crop_in = layers.Input(shape=(1,), name="crop_input")
    soil_emb = layers.Flatten()(layers.Embedding(n_soil, max(2, n_soil // 2))(soil_in))
    crop_emb = layers.Flatten()(layers.Embedding(n_crop, max(2, n_crop // 2))(crop_in))
    x = layers.Concatenate()([num_in, soil_emb, crop_emb])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(len(fert_encoder.classes_), activation="softmax", name="fert_output")(x)
    model = keras.Model([num_in, soil_in, crop_in], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit([Xn_tr, Xs_tr, Xc_tr], y_tr, validation_split=0.1, epochs=40, batch_size=16, verbose=0)

    acc = accuracy_score(y_te, model.predict([Xn_te, Xs_te, Xc_te], verbose=0).argmax(axis=1))
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model.save(models_dir / "fert_model.keras")
    with open(models_dir / "fert_preprocessors.pkl", "wb") as fh:
        pickle.dump({"fert_crop_encoder": fert_crop_encoder, "soil_encoder": soil_encoder,
                     "fert_encoder": fert_encoder, "scaler_fert": scaler_fert,
                     "num_cols_fert": NUM_COLS_FERT}, fh)
    print(json.dumps({"rows": len(df), "params": model.count_params(), "n_test": int(len(y_te)),
                      "test_accuracy": float(acc), "fert_classes": fert_encoder.classes_.tolist(),
                      "fert_crops": fert_crop_encoder.classes_.tolist(),
                      "soil_types": soil_encoder.classes_.tolist()}, indent=2))


if __name__ == "__main__":
    main()
