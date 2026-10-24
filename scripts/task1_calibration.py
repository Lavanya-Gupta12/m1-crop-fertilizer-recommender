"""Task 1 - calibration of the stage-1 crop model.

Usage:
    python scripts/task1_calibration.py baseline

`baseline` (step 1, no fixes):
  * leakage check: notebook CV protocol with the scaler fit on all rows vs inside each fold
  * out-of-fold calibration of the unchanged architecture (acc, macro-F1, ECE, NLL, Brier)
  * out-of-range probes: top-1 probability on zero / extreme / impossible inputs
Outputs go to results/task1/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

from m1.calibration import all_metrics, plot_reliability  # noqa: E402

SEED = 42
N_FOLDS = 5
EPOCHS = 40          # one epoch policy for every Task-1 variant (tuning is Task 3)
BATCH = 16
CAL_FRACTION = 0.15  # held out of each training fold for temperature scaling
NUM_COLS_CROP = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
OUT = ROOT / "results" / "task1"


# ------------------------------------------------------------------ setup
def set_seeds(seed):
    keras.utils.set_random_seed(seed)
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()


def load_crop_df():
    local = ROOT / "data" / "Crop_recommendation.csv"
    if local.exists():
        path = local
    else:
        import kagglehub
        path = Path(kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")) / "Crop_recommendation.csv"
    df = pd.read_csv(path).dropna().drop_duplicates()
    df["label"] = df["label"].map(lambda s: str(s).strip().lower().replace(" ", ""))
    return df


def build_crop_model(n_numeric, n_classes):
    """Same architecture as the notebook; the softmax is split into its own layer so the
    pre-softmax logits are available for temperature scaling (identical maths and params)."""
    num_in = layers.Input(shape=(n_numeric,), name="numeric_input")
    x = layers.Dense(64, activation="relu")(num_in)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    logits = layers.Dense(n_classes, name="logits")(x)
    out = layers.Activation("softmax", name="crop_output")(logits)
    model = keras.Model(num_in, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def predict_logits(model, X):
    logit_model = keras.Model(model.inputs, model.get_layer("logits").output)
    return np.asarray(logit_model.predict(X, verbose=0), dtype=np.float64)


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def summarise(per_fold):
    keys = per_fold[0].keys()
    return {k: {"mean": float(np.mean([f[k] for f in per_fold])),
                "std": float(np.std([f[k] for f in per_fold], ddof=1))} for k in keys}


# ------------------------------------------------------------ probes
def make_probes(df):
    X = df[NUM_COLS_CROP]
    med, mx = X.median(), X.max()
    probes = {
        "rice_like (in-distribution)": [90, 42, 43, 20.8, 82, 6.5, 202],
        "all zeros": [0] * 7,
        "all features 2x train max": (2 * mx).tolist(),
        "all features 5x train max": (5 * mx).tolist(),
    }
    for col in NUM_COLS_CROP:
        row = med.copy(); row[col] = 3 * mx[col]
        probes[f"{col} = 3x train max, rest median"] = row.tolist()
    for name, col, val in [("pH = 0 (impossible)", "ph", 0.0), ("pH = 14 (impossible)", "ph", 14.0),
                           ("temperature = 60 C", "temperature", 60.0),
                           ("humidity = 150 % (impossible)", "humidity", 150.0),
                           ("rainfall = 3000 mm", "rainfall", 3000.0)]:
        row = med.copy(); row[col] = val
        probes[name] = row.tolist()
    return probes


# ------------------------------------------------------------ baseline
def leakage_check(X, y, n_classes):
    """Reproduce the notebook's CV (25 epochs, full training fold) with the scaler fit
    (a) on all rows before CV, as in the notebook, and (b) inside each training fold."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scaler_all = StandardScaler().fit(X)
    res = {"leaky_scaler_on_all_rows": [], "scaler_inside_fold": []}
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        for key in res:
            sc = scaler_all if key.startswith("leaky") else StandardScaler().fit(X[tr])
            set_seeds(SEED + fold)
            m = build_crop_model(X.shape[1], n_classes)
            m.fit(sc.transform(X[tr]), y[tr], epochs=25, batch_size=BATCH, verbose=0)
            p = np.asarray(m.predict(sc.transform(X[te]), verbose=0))
            mt = all_metrics(p, y[te])
            res[key].append({"accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"]})
        print(f"  leakage fold {fold + 1}: leaky acc={res['leaky_scaler_on_all_rows'][-1]['accuracy']:.4f} "
              f"| in-fold acc={res['scaler_inside_fold'][-1]['accuracy']:.4f}")
    return {k: {"per_fold": v, "summary": summarise(v)} for k, v in res.items()}


def run_baseline():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_crop_df()
    enc = LabelEncoder().fit(sorted(df["label"].unique()))
    X = df[NUM_COLS_CROP].values.astype(float)
    y = enc.transform(df["label"])
    n_classes = len(enc.classes_)
    probes = make_probes(df)
    probe_X = np.array(list(probes.values()), dtype=float)
    print(f"backend={keras.backend.backend()} keras={keras.__version__} rows={len(X)} classes={n_classes}")

    t0 = time.time()
    print("[1/2] leakage check (notebook protocol, 25 epochs)")
    leakage = leakage_check(X, y, n_classes)

    print(f"[2/2] baseline calibration ({N_FOLDS}-fold, {EPOCHS} epochs, "
          f"{int(CAL_FRACTION * 100)}% of each training fold held out for later calibration)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_logits = np.zeros((len(X), n_classes))
    fold_of = np.zeros(len(X), dtype=int)
    per_fold, prior_per_fold, cal_store, probe_probs = [], [], {}, []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        fit_idx, cal_idx = train_test_split(tr, test_size=CAL_FRACTION, stratify=y[tr],
                                            random_state=SEED + fold)
        scaler = StandardScaler().fit(X[fit_idx])          # fit on training rows only
        set_seeds(SEED + fold)
        m = build_crop_model(X.shape[1], n_classes)
        m.fit(scaler.transform(X[fit_idx]), y[fit_idx], epochs=EPOCHS, batch_size=BATCH, verbose=0)

        z_te = predict_logits(m, scaler.transform(X[te]))
        oof_logits[te] = z_te
        fold_of[te] = fold
        per_fold.append(all_metrics(softmax_np(z_te), y[te]))
        cal_store[f"fold{fold}_cal_idx"] = cal_idx
        cal_store[f"fold{fold}_cal_logits"] = predict_logits(m, scaler.transform(X[cal_idx]))
        probe_probs.append(softmax_np(predict_logits(m, scaler.transform(probe_X))))

        # reference: predict the training-fold class frequencies for every row
        prior = np.bincount(y[fit_idx], minlength=n_classes) / len(fit_idx)
        prior_per_fold.append(all_metrics(np.tile(prior, (len(te), 1)), y[te]))
        print(f"  fold {fold + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in per_fold[-1].items()))

    oof_probs = softmax_np(oof_logits)
    pooled = all_metrics(oof_probs, y)
    conf = oof_probs.max(axis=1)
    correct = oof_probs.argmax(axis=1) == y

    # probes: top-1 prob/class per fold model
    probe_probs = np.stack(probe_probs)                     # (folds, probes, classes)
    probe_rows = []
    for i, name in enumerate(probes):
        top = probe_probs[:, i, :].max(axis=1)
        cls = enc.classes_[probe_probs[:, i, :].argmax(axis=1)]
        probe_rows.append({"probe": name, "input": [round(v, 2) for v in probe_X[i]],
                           "top1_prob_min": float(top.min()), "top1_prob_median": float(np.median(top)),
                           "top1_prob_max": float(top.max()),
                           "predicted_classes_across_folds": sorted(set(cls.tolist()))})

    results = {
        "config": {"backend": keras.backend.backend(), "keras": keras.__version__, "seed": SEED,
                   "folds": N_FOLDS, "epochs": EPOCHS, "batch": BATCH, "cal_fraction": CAL_FRACTION,
                   "ece_bins": 15, "rows": int(len(X)), "classes": int(n_classes)},
        "leakage_check": leakage,
        "baseline_per_fold": per_fold,
        "baseline_summary": summarise(per_fold),
        "baseline_pooled_oof": pooled,
        "prior_reference_summary": summarise(prior_per_fold),
        "confidence_profile": {
            "share_conf_ge_0.99": float(np.mean(conf >= 0.99)),
            "share_conf_ge_0.999": float(np.mean(conf >= 0.999)),
            "accuracy_when_conf_ge_0.99": float(correct[conf >= 0.99].mean()),
            "n_errors": int((~correct).sum()),
            "median_conf_on_errors": float(np.median(conf[~correct])),
            "share_errors_with_conf_ge_0.9": float(np.mean(conf[~correct] >= 0.9)),
        },
        "probes": probe_rows,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (OUT / "baseline.json").write_text(json.dumps(results, indent=2))
    np.savez(OUT / "baseline_oof.npz", oof_logits=oof_logits, y=y, fold_of=fold_of,
             classes=enc.classes_, **cal_store)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_reliability(oof_probs, y, axes[0], title="Stage 1 baseline - out-of-fold reliability")
    axes[1].hist([conf[correct], conf[~correct]], bins=np.linspace(0, 1, 21), stacked=True,
                 label=["correct", "wrong"], color=["C0", "C3"])
    axes[1].set_yscale("log"); axes[1].set_xlabel("top-1 probability"); axes[1].set_ylabel("count (log)")
    axes[1].set_title("Confidence histogram (out-of-fold)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "baseline_reliability.png", dpi=150)
    print(f"done in {results['runtime_seconds']}s -> {OUT}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline"])
    args = ap.parse_args()
    {"baseline": run_baseline}[args.stage]()
