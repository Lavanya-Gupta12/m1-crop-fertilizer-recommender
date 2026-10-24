"""Train and save the deployed stage-2 fertilizer model.

Model: the embedding NN (notebook architecture; project scope is deep learning). The
RandomForest scored higher in the benchmark but within noise and is kept as a benchmark result
only (see docs/architecture.md). Preprocessing (scaler, label encoders) is fit inside FertNN.fit
on its own training data. The deployed model is trained on all 99 rows; its performance estimate
is the repeated 5-fold CV in scripts/benchmark.py (preprocessing fit inside each fold).

Also saves an input-range guard over the fertilizer data's numeric features: stage-1 inputs
usually lie outside them (only 1.7% of Tier-1/2 crop rows are inside on all shared features).

Usage: python scripts/train_stage2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

from m1.calibration import InputGuard, all_metrics  # noqa: E402
from m1.stage2 import CAT_COLS_FERT, NUM_COLS_FERT, FertNN  # noqa: E402
from task1_calibration import SEED  # noqa: E402
from train_stage2_notebook import load_fert_df  # noqa: E402


def main():
    df = load_fert_df().reset_index(drop=True)
    classes = sorted(df["fertilizer"].unique())
    y = np.array([classes.index(v) for v in df["fertilizer"]])
    soils, crops = sorted(df["soil_type"].unique()), sorted(df["label"].unique())

    # check: the moved FertNN code reproduces benchmark split 1 exactly
    tr, te = next(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(df, y))
    m = FertNN(False, SEED, soils, crops, len(classes)).fit(df.iloc[tr], y[tr])
    check = all_metrics(m.predict_proba(df.iloc[te]), y[te])
    logged = json.loads((ROOT / "results" / "benchmark" / "benchmark.json").read_text())
    logged = logged["stage2"]["Embedding NN (current)"]["per_fold"][0]
    reproduces = all(abs(check[k] - logged[k]) < 1e-9 for k in ("accuracy", "macro_f1", "nll", "ece"))
    print("benchmark split 1 reproduced:", reproduces, {k: round(check[k], 4) for k in ("accuracy", "nll")})

    # deployed model: all 99 rows
    model = FertNN(False, SEED, soils, crops, len(classes)).fit(df[NUM_COLS_FERT + CAT_COLS_FERT], y)
    guard = InputGuard(NUM_COLS_FERT, margin=0.10).fit(df[NUM_COLS_FERT].values)
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    for old in ("fert_rf.joblib", "fert_model.keras", "fert_preprocessors.pkl"):
        (models / old).unlink(missing_ok=True)
    model.save(models)
    config = {"kind": "keras_embedding_nn", "model_stem": "fert_nn",
              "model": "numeric(6, standardised) + soil Embedding(5->2) + crop Embedding(11->5) -> Dense64 -> "
                       "Dropout0.2 -> Dense32 -> Softmax7; Adam, batch 16, 40 epochs, seed 42",
              "params": model.count_params(),
              "trained_on": "all 99 rows; estimate = repeated stratified 5-fold x3 (results/benchmark)",
              "num_cols": NUM_COLS_FERT, "cat_cols": CAT_COLS_FERT, "classes": classes,
              "soil_types": soils, "crops": crops, "guard": guard.to_dict(),
              "cv_estimate": "accuracy 0.9330 +/- 0.0407, macro-F1 0.8451 +/- 0.0835, ECE 0.1839 (15 splits)",
              "confidence_note": "Uncalibrated; under-confident in CV (mean confidence 0.80 vs accuracy 0.93)."}
    (models / "fert_config.json").write_text(json.dumps(config, indent=2))
    train_acc = float((model.predict_proba(df).argmax(1) == y).mean())
    print(json.dumps({"rows": len(df), "params": model.count_params(), "classes": classes,
                      "train_accuracy_not_an_estimate": train_acc,
                      "benchmark_split1_reproduced": reproduces}, indent=2))


if __name__ == "__main__":
    main()
