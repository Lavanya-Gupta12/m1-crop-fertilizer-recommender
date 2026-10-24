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


def build_crop_model(n_numeric, n_classes, label_smoothing=0.0):
    """Same architecture as the notebook; the softmax is split into its own layer so the
    pre-softmax logits are available for temperature scaling (identical maths and params).
    label_smoothing > 0 switches to one-hot targets + CategoricalCrossentropy."""
    num_in = layers.Input(shape=(n_numeric,), name="numeric_input")
    x = layers.Dense(64, activation="relu")(num_in)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    logits = layers.Dense(n_classes, name="logits")(x)
    out = layers.Activation("softmax", name="crop_output")(logits)
    model = keras.Model(num_in, out)
    if label_smoothing > 0:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    else:
        loss = "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def train_crop_model(X, y, n_classes, seed, label_smoothing=0.0):
    set_seeds(seed)
    m = build_crop_model(X.shape[1], n_classes, label_smoothing)
    target = keras.utils.to_categorical(y, n_classes) if label_smoothing > 0 else y
    m.fit(X, target, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    return m


def fold_splits(y):
    """The fixed Task-1 folds: (fold, fit_idx, cal_idx, test_idx), identical to `baseline`."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        fit_idx, cal_idx = train_test_split(tr, test_size=CAL_FRACTION, stratify=y[tr],
                                            random_state=SEED + fold)
        yield fold, fit_idx, cal_idx, te


def load_xy():
    df = load_crop_df()
    enc = LabelEncoder().fit(sorted(df["label"].unique()))
    return df, df[NUM_COLS_CROP].values.astype(float), enc.transform(df["label"]), enc


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


# ------------------------------------------------------------ variants A-E
LS_VALUES = (0.05, 0.1)
N_ENSEMBLE = 5


def ensemble_seed(fold, member):
    return SEED + fold + 100 * member      # member 0 == the variant-A model


def mean_prob_logits(logit_list):
    """Ensemble = average of member probabilities; returned as log-probabilities so that
    temperature scaling can be applied to the ensemble exactly like to a single model."""
    return np.log(np.clip(np.mean([softmax_np(z) for z in logit_list], axis=0), 1e-12, 1.0))


def run_variants():
    from m1.calibration import TemperatureScaler, nll as nll_fn
    df, X, y, enc = load_xy()
    n_classes = len(enc.classes_)
    probes = make_probes(df)
    probe_X = np.array(list(probes.values()), dtype=float)
    base = np.load(OUT / "baseline_oof.npz", allow_pickle=True)
    print(f"backend={keras.backend.backend()} rows={len(X)} classes={n_classes}")
    t0 = time.time()

    # raw logits per model family: key -> (oof (n,c), {fold: cal logits}, {fold: probe logits})
    fams = {k: [np.zeros((len(X), n_classes)), {}, {}] for k in
            ["A"] + [f"B_ls{ls}" for ls in LS_VALUES] + ["E_raw"]}
    cal_idx_of = {}
    for fold, fit_idx, cal_idx, te in fold_splits(y):
        cal_idx_of[fold] = cal_idx
        scaler = StandardScaler().fit(X[fit_idx])
        Xf, Xc, Xt, Xp = (scaler.transform(a) for a in (X[fit_idx], X[cal_idx], X[te], probe_X))
        members = []
        for member in range(N_ENSEMBLE):
            m = train_crop_model(Xf, y[fit_idx], n_classes, ensemble_seed(fold, member))
            members.append([predict_logits(m, a) for a in (Xt, Xc, Xp)])
        for key, logit_triplet in [("A", members[0]),
                                   ("E_raw", [mean_prob_logits([mm[i] for mm in members]) for i in range(3)])]:
            fams[key][0][te] = logit_triplet[0]; fams[key][1][fold] = logit_triplet[1]; fams[key][2][fold] = logit_triplet[2]
        for ls in LS_VALUES:
            m = train_crop_model(Xf, y[fit_idx], n_classes, SEED + fold, label_smoothing=ls)
            k = f"B_ls{ls}"
            fams[k][0][te] = predict_logits(m, Xt); fams[k][1][fold] = predict_logits(m, Xc)
            fams[k][2][fold] = predict_logits(m, Xp)
        print(f"  fold {fold + 1} trained ({time.time() - t0:.0f}s)")

    a_matches = bool(np.array_equal(fams["A"][0], base["oof_logits"]))
    print("variant A reproduces baseline OOF logits exactly:", a_matches)

    # D: choose the label-smoothing value by calibration-split NLL (never the test folds)
    cal_nll = {ls: float(np.mean([nll_fn(softmax_np(fams[f"B_ls{ls}"][1][f]), y[cal_idx_of[f]])
                                  for f in range(N_FOLDS)])) for ls in LS_VALUES}
    best_ls = min(cal_nll, key=cal_nll.get)
    print("label smoothing: mean cal-split NLL", cal_nll, "-> D uses ls =", best_ls)

    # assemble variants; temperature fitted per fold on that fold's calibration split only
    specs = {"A_baseline": ("A", False), **{f"B_ls{ls}": (f"B_ls{ls}", False) for ls in LS_VALUES},
             "C_A+TS": ("A", True), f"D_ls{best_ls}+TS": (f"B_ls{best_ls}", True),
             "E_ensemble5": ("E_raw", False), "E_ensemble5+TS": ("E_raw", True)}
    results, oof_store = {}, {}
    for name, (fam, use_ts) in specs.items():
        oof_logits, cal_logits, probe_logits = fams[fam]
        oof_probs = np.zeros_like(oof_logits)
        temps, per_fold, probe_top = [], [], []
        for fold, _, cal_idx, te in fold_splits(y):
            T = TemperatureScaler().fit(cal_logits[fold], y[cal_idx]).temperature if use_ts else 1.0
            temps.append(T)
            oof_probs[te] = softmax_np(oof_logits[te] / T)
            per_fold.append(all_metrics(oof_probs[te], y[te]))
            probe_top.append(softmax_np(probe_logits[fold] / T).max(axis=1))
        probe_top = np.stack(probe_top)
        results[name] = {"temperatures": temps, "per_fold": per_fold, "summary": summarise(per_fold),
                         "pooled_oof": all_metrics(oof_probs, y),
                         "probe_top1_median": {p: float(np.median(probe_top[:, i]))
                                               for i, p in enumerate(probes)}}
        oof_store[name] = oof_probs
        s = results[name]["summary"]
        print(f"  {name:18s} acc={s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f} "
              f"ece={s['ece']['mean']:.4f}±{s['ece']['std']:.4f} nll={s['nll']['mean']:.4f}±{s['nll']['std']:.4f} "
              f"T={np.round(temps, 3).tolist()}")

    out = {"config": {"backend": keras.backend.backend(), "epochs": EPOCHS, "batch": BATCH,
                      "cal_fraction": CAL_FRACTION, "ls_values": LS_VALUES, "n_ensemble": N_ENSEMBLE,
                      "ensemble_seeds": "SEED + fold + 100*member (member 0 = variant A)"},
           "variant_A_reproduces_baseline": a_matches, "ls_selection_cal_nll": cal_nll,
           "best_ls": best_ls, "variants": results, "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "variants.json").write_text(json.dumps(out, indent=2))
    np.savez(OUT / "variants_oof.npz", y=y, **{k.replace("+", "_plus_"): v for k, v in oof_store.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, (name, probs) in zip(axes.ravel(), oof_store.items()):
        plot_reliability(probs, y, ax, title=name)
    for ax in axes.ravel()[len(oof_store):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "variants_reliability.png", dpi=120)
    print(f"done in {out['runtime_seconds']}s")


# ------------------------------------------------------------ final model
TARGET_SELECTIVE_ACC = 0.99   # threshold policy: smallest t whose accepted set reaches this accuracy


def coverage_curve(probs, y, thresholds=np.round(np.arange(0.30, 1.0, 0.05), 2)):
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    rows = []
    for t in thresholds:
        acc_mask = conf >= t
        rows.append({"threshold": float(t), "coverage": float(acc_mask.mean()),
                     "accuracy_accepted": float(correct[acc_mask].mean()) if acc_mask.any() else None,
                     "accuracy_flagged": float(correct[~acc_mask].mean()) if (~acc_mask).any() else None,
                     "n_flagged": int((~acc_mask).sum())})
    return rows


def run_final(variant):
    import pickle
    from m1.calibration import InputGuard, TemperatureScaler
    MODELS = ROOT / "models"
    MODELS.mkdir(exist_ok=True)
    df, X, y, enc = load_xy()
    n_classes = len(enc.classes_)
    ensemble = variant.startswith("E")
    use_ts = variant.endswith("+TS")
    ls = float(variant.split("_ls")[1].split("+")[0]) if "_ls" in variant else 0.0

    # threshold from the out-of-fold coverage-vs-accuracy curve of the chosen variant
    oof = np.load(OUT / "variants_oof.npz")
    curve = coverage_curve(oof[variant.replace("+", "_plus_")], oof["y"])
    ok = [r for r in curve if r["accuracy_accepted"] is not None and r["accuracy_accepted"] >= TARGET_SELECTIVE_ACC]
    threshold = min(r["threshold"] for r in ok)

    # same 80/20 split as the notebook; 15% of the 80% held out to fit T
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    fit_idx, cal_idx = train_test_split(tr, test_size=CAL_FRACTION, stratify=y[tr], random_state=SEED)
    scaler = StandardScaler().fit(X[fit_idx])
    Xf, Xc, Xt = (scaler.transform(a) for a in (X[fit_idx], X[cal_idx], X[te]))
    n_members = N_ENSEMBLE if ensemble else 1
    models = [train_crop_model(Xf, y[fit_idx], n_classes, SEED + 100 * k, label_smoothing=ls)
              for k in range(n_members)]
    zs = {name: [predict_logits(m, a) for m in models] for name, a in (("cal", Xc), ("test", Xt))}
    agg = (lambda L: mean_prob_logits(L)) if ensemble else (lambda L: L[0])
    T = TemperatureScaler().fit(agg(zs["cal"]), y[cal_idx]).temperature if use_ts else 1.0
    test_probs = softmax_np(agg(zs["test"]) / T)
    single_raw = softmax_np(zs["test"][0])           # = variant A trained on the same data
    guard = InputGuard(NUM_COLS_CROP, margin=0.10).fit(X[tr])
    conf = test_probs.max(axis=1)
    test_res = {"variant": variant, "temperature": T, "threshold": threshold,
                "n_test": int(len(te)), "test_metrics": all_metrics(test_probs, y[te]),
                "test_metrics_A_same_split": all_metrics(single_raw, y[te]),
                "test_at_threshold": {"coverage": float((conf >= threshold).mean()),
                                      "accuracy_accepted": float((test_probs.argmax(1) == y[te])[conf >= threshold].mean()),
                                      "n_flagged_uncertain": int((conf < threshold).sum())},
                "oof_coverage_curve": curve, "target_selective_accuracy": TARGET_SELECTIVE_ACC,
                "guard": guard.to_dict()}

    for f in MODELS.glob("crop_model*.keras"):
        f.unlink()
    names = ["crop_model.keras"] if n_members == 1 else [f"crop_model_{k}.keras" for k in range(n_members)]
    for m, nm in zip(models, names):
        m.save(MODELS / nm)
    config = {"variant": variant, "model_files": names, "aggregate": "mean_prob" if ensemble else "single",
              "temperature": T, "uncertain_threshold": threshold, "feature_names": NUM_COLS_CROP,
              "classes": enc.classes_.tolist(), "guard": guard.to_dict(),
              "trained_on": "80% stratified split (random_state=42), fit on 85% of it, T fitted on the other 15%",
              "backend": keras.backend.backend(), "keras": keras.__version__}
    (MODELS / "crop_config.json").write_text(json.dumps(config, indent=2))
    with open(MODELS / "crop_preprocessors.pkl", "wb") as fh:
        pickle.dump({"scaler_crop": scaler, "crop_encoder": enc}, fh)
    (OUT / "final.json").write_text(json.dumps(test_res, indent=2))
    print(json.dumps({k: v for k, v in test_res.items() if k not in ("oof_coverage_curve", "guard")}, indent=2))
    print("coverage curve (OOF):")
    for r in curve:
        print(f"  t={r['threshold']:.2f} coverage={r['coverage']:.4f} acc_accepted={r['accuracy_accepted']:.4f} "
              f"acc_flagged={r['accuracy_flagged']} n_flagged={r['n_flagged']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "variants", "final"])
    ap.add_argument("--variant", help="final stage: variant name from variants.json, e.g. 'E_ensemble5+TS'")
    args = ap.parse_args()
    if args.stage == "final":
        run_final(args.variant)
    else:
        {"baseline": run_baseline, "variants": run_variants}[args.stage]()
