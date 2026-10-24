"""Train and save the deployed stage-2 fertilizer model (step 6).

Model: the stage-2 benchmark winner on accuracy and macro-F1 - RandomForest (300 trees) on scaled
numerics + one-hot soil type and crop, as one sklearn Pipeline. All preprocessing is fit
inside the pipeline on its training data. The deployed model is trained on all 99 rows; its
performance estimate is the repeated 5-fold CV in scripts/benchmark.py (preprocessing fit
inside each fold), not a held-out split.

Also saves an input-range guard over the fertilizer data's numeric features: stage-1 inputs
can lie far outside them (e.g. humidity 14-100% in the crop data vs 50-72% here).

Usage: python scripts/train_stage2.py
"""
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import CAT_COLS_FERT, make_rf_pipeline  # noqa: E402
from m1.calibration import InputGuard  # noqa: E402
from task1_calibration import SEED  # noqa: E402
from train_stage2_notebook import NUM_COLS_FERT, load_fert_df  # noqa: E402


def main():
    df = load_fert_df().reset_index(drop=True)
    pipe = make_rf_pipeline(SEED).fit(df[NUM_COLS_FERT + CAT_COLS_FERT], df["fertilizer"])
    guard = InputGuard(NUM_COLS_FERT, margin=0.10).fit(df[NUM_COLS_FERT].values)
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(pipe, models / "fert_rf.joblib")
    config = {"kind": "sklearn_pipeline", "model_file": "fert_rf.joblib",
              "model": "RandomForest(300 trees, random_state=42) on StandardScaler(numerics) + OneHot(soil_type, crop)",
              "trained_on": "all 99 rows; estimate = repeated stratified 5-fold x3 (results/benchmark)",
              "num_cols": NUM_COLS_FERT, "cat_cols": CAT_COLS_FERT,
              "classes": pipe.classes_.tolist(), "soil_types": sorted(df["soil_type"].unique()),
              "crops": sorted(df["label"].unique()), "guard": guard.to_dict(),
              "cv_estimate": "accuracy 0.9667 +/- 0.0408, macro-F1 0.9266 +/- 0.0908 (15 splits)",
              "confidence_note": "Uncalibrated and strongly under-confident (CV mean confidence 0.64 vs accuracy 0.97)."}
    (models / "fert_config.json").write_text(json.dumps(config, indent=2))
    train_acc = float((pipe.predict(df[NUM_COLS_FERT + CAT_COLS_FERT]) == df["fertilizer"]).mean())
    print(json.dumps({"rows": len(df), "classes": config["classes"], "train_accuracy_not_an_estimate": train_acc,
                      "guard_train_min": dict(zip(NUM_COLS_FERT, guard.train_min_.tolist())),
                      "guard_train_max": dict(zip(NUM_COLS_FERT, guard.train_max_.tolist()))}, indent=2))


if __name__ == "__main__":
    main()
