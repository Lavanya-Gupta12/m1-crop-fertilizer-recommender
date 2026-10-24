"""Tasks 3+4 - benchmark both stages under one protocol.

Stage 1: stratified 5-fold (same outer folds as Task 1), scaler inside each fold, seed 42.
         LogisticRegression, RandomForest, LightGBM, current MLP (40 epochs, full training fold).
Stage 2: repeated stratified 5-fold x 3 (n=99), preprocessing inside each fold.
         Majority class, RandomForest (one-hot), current embedding NN, one-hot NN.
Metrics (mean +/- std over folds, ddof=1): accuracy, macro-F1, ECE, NLL, parameter count, train time.

Usage: python scripts/benchmark.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import keras  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # noqa: E402

from m1.calibration import all_metrics  # noqa: E402
from task1_calibration import BATCH, EPOCHS, SEED, build_crop_model, load_xy, set_seeds  # noqa: E402
from m1.stage2 import CAT_COLS_FERT, NUM_COLS_FERT, FertNN  # noqa: E402
from train_stage2_notebook import load_fert_df  # noqa: E402

OUT = ROOT / "results" / "benchmark"


def summarise(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1))}
            for k in rows[0]}


def tree_nodes(model):
    """Size of a tree ensemble = total number of nodes (the closest analogue of a parameter count)."""
    if isinstance(model, RandomForestClassifier):
        return int(sum(e.tree_.node_count for e in model.estimators_))
    info = model.booster_.dump_model()["tree_info"]          # LightGBM
    return int(sum(2 * t["num_leaves"] - 1 for t in info))


# ------------------------------------------------------------------ stage 1
def stage1():
    from lightgbm import LGBMClassifier
    _, X, y, enc = load_xy()
    n_classes = len(enc.classes_)
    sk_models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=5000, random_state=SEED),
        "RandomForest (300 trees)": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "LightGBM (defaults)": lambda: LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=4),
    }
    res = {name: [] for name in [*sk_models, "MLP 64-32 (current)"]}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        for name, make in sk_models.items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", make())])
            t0 = time.perf_counter(); pipe.fit(X[tr], y[tr]); t_train = time.perf_counter() - t0
            t0 = time.perf_counter(); p = pipe.predict_proba(X[te]); t_inf = time.perf_counter() - t0
            clf = pipe.named_steps["clf"]
            size = clf.coef_.size + clf.intercept_.size if name == "LogisticRegression" else tree_nodes(clf)
            res[name].append({**all_metrics(p, y[te]), "params": size, "train_s": t_train,
                              "infer_ms_per_row": 1000 * t_inf / len(te)})
        scaler = StandardScaler().fit(X[tr])
        set_seeds(SEED + fold)
        m = build_crop_model(X.shape[1], n_classes)
        t0 = time.perf_counter()
        m.fit(scaler.transform(X[tr]), y[tr], epochs=EPOCHS, batch_size=BATCH, verbose=0)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter(); p = np.asarray(m.predict(scaler.transform(X[te]), verbose=0), dtype=float)
        t_inf = time.perf_counter() - t0
        res["MLP 64-32 (current)"].append({**all_metrics(p, y[te]), "params": m.count_params(),
                                           "train_s": t_train, "infer_ms_per_row": 1000 * t_inf / len(te)})
        print(f"  stage1 fold {fold + 1}: " + " | ".join(f"{k} acc={v[-1]['accuracy']:.4f}" for k, v in res.items()))
    return {k: {"per_fold": v, "summary": summarise(v)} for k, v in res.items()}


# ------------------------------------------------------------------ stage 2
def make_rf_pipeline(seed):
    pre = ColumnTransformer([("num", StandardScaler(), NUM_COLS_FERT),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS_FERT)])
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))])


def stage2():
    df = load_fert_df().reset_index(drop=True)
    fert_enc = LabelEncoder().fit(sorted(df["fertilizer"].unique()))
    y = fert_enc.transform(df["fertilizer"])
    n_classes = len(fert_enc.classes_)
    soils, crops = sorted(df["soil_type"].unique()), sorted(df["label"].unique())
    names = ["Majority class", "RandomForest (one-hot, 300 trees)", "Embedding NN (current)", "One-hot NN"]
    res = {n: [] for n in names}
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    for k, (tr, te) in enumerate(rskf.split(df, y)):
        dtr, dte = df.iloc[tr], df.iloc[te]
        models = {
            "Majority class": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
            "RandomForest (one-hot, 300 trees)": make_rf_pipeline(SEED + k),
            "Embedding NN (current)": FertNN(False, SEED + k, soils, crops, n_classes),
            "One-hot NN": FertNN(True, SEED + k, soils, crops, n_classes),
        }
        for name, m in models.items():
            t0 = time.perf_counter(); m.fit(dtr, y[tr]); t_train = time.perf_counter() - t0
            p = m.predict_proba(dte)
            if name == "Majority class":
                full = np.zeros((len(te), n_classes)); full[:, m.named_steps["clf"].classes_] = p; p = full
                size = 0
            elif name.startswith("RandomForest"):
                size = tree_nodes(m.named_steps["clf"])
            else:
                size = m.count_params()
            res[name].append({**all_metrics(p, y[te]), "params": size, "train_s": t_train})
        print(f"  stage2 split {k + 1}/15: " + " | ".join(f"{n.split(' ')[0]} acc={v[-1]['accuracy']:.3f}" for n, v in res.items()))
    return {k: {"per_fold": v, "summary": summarise(v)} for k, v in res.items()}


def table(results, cols):
    lines = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in results.items():
        s = r["summary"]
        cells = []
        for c in cols:
            mean, std = s[c]["mean"], s[c]["std"]
            cells.append(f"{mean:,.0f}" if c == "params" else
                         f"{mean:.2f} ± {std:.2f}" if c in ("train_s",) else
                         f"{mean:.3f} ± {std:.3f}" if c == "infer_ms_per_row" else f"{mean:.4f} ± {std:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("stage 1 (5-fold)"); s1 = stage1()
    print("stage 2 (repeated 5-fold x 3)"); s2 = stage2()
    out = {"config": {"backend": keras.backend.backend(), "keras": keras.__version__, "seed": SEED,
                      "nn_epochs": EPOCHS, "batch": BATCH, "ece_bins": 15},
           "stage1": s1, "stage2": s2, "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "benchmark.json").write_text(json.dumps(out, indent=2))
    cols = ["accuracy", "macro_f1", "ece", "nll", "params", "train_s"]
    md = ("## Stage 1\n\n" + table(s1, cols + ["infer_ms_per_row"]) +
          "\n\n## Stage 2\n\n" + table(s2, cols) + "\n")
    (OUT / "benchmark_tables.md").write_text(md)
    print(md); print(f"done in {out['runtime_seconds']}s")
