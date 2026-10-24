"""SHAP summaries for the deployed models.

Crop model: KernelExplainer on the deployed calibrated ensemble (probabilities, raw input units),
            background = k-means(20) of the 80% training split, explaining 220 test rows
            (10 per class) for all 22 classes.
Fertilizer: KernelExplainer on the deployed embedding NN, inputs = 6 numerics + soil type + crop
            (each categorical is one feature), background = 20 real rows, all 99 rows explained
            (= its training data).
Also saves models/shap_background.npz used by the app for per-prediction reasons.

Usage: python scripts/shap_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from m1.recommender import get_recommender  # noqa: E402
from task1_calibration import NUM_COLS_CROP, SEED, load_xy  # noqa: E402
from train_stage2_notebook import load_fert_df  # noqa: E402

OUT = ROOT / "results" / "shap"


def stacked_bar(mean_abs, feature_names, class_names, title, path):
    """mean_abs: (features, classes). Stacked horizontal bars, features sorted by total."""
    order = np.argsort(mean_abs.sum(axis=1))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(feature_names) + 2.5))
    left = np.zeros(len(order))
    # distinct colour per class: tab20 + every 4th colour of tab20b covers up to 25 classes
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)[::4]
    for c, name in enumerate(class_names):
        vals = mean_abs[order, c]
        ax.barh(np.array(feature_names)[order], vals, left=left, color=colors[c], label=name)
        left += vals
    ax.set_xlabel("mean |SHAP value| (probability units), stacked over classes")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2 if len(class_names) > 10 else 1, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = get_recommender()
    df, X, y, enc = load_xy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)

    # ---------------- crop model (deployed ensemble + TS)
    bg = shap.kmeans(X[tr], 20)
    rng = np.random.default_rng(SEED)
    sample = np.concatenate([rng.choice(te[y[te] == c], 10, replace=False) for c in range(len(enc.classes_))])
    explainer = shap.KernelExplainer(rec.crop_probs, bg)
    sv = np.asarray(explainer.shap_values(X[sample], nsamples=256, silent=True))   # (n, features, classes)
    crop_mean_abs = np.abs(sv).mean(axis=0)                                          # (features, classes)
    stacked_bar(crop_mean_abs, NUM_COLS_CROP, enc.classes_.tolist(),
                "Crop model (deployed ensemble + TS): mean |SHAP| over 220 test rows, all 22 classes",
                OUT / "crop_shap_mean_abs.png")

    # ---------------- fertilizer model (deployed embedding NN)
    fdf = load_fert_df().reset_index(drop=True)
    cfg = rec.fert_cfg
    Zf = rec.fert_encode(fdf)                                  # [6 numerics, soil id, crop id]
    fert_bg = Zf[rng.choice(len(Zf), 20, replace=False)]       # real rows: valid categorical values
    fsv = np.asarray(shap.KernelExplainer(rec.fert_probs_encoded, fert_bg).shap_values(
        Zf, nsamples=256, silent=True))                        # (n, 8, classes); 2^8 = 256 -> exact enumeration
    fert_mean_abs = np.abs(fsv).mean(axis=0)
    groups = rec.fert_feature_names
    stacked_bar(fert_mean_abs, groups, cfg["classes"],
                "Fertilizer model (deployed embedding NN): mean |SHAP| over 99 rows, all 7 classes",
                OUT / "fert_shap_mean_abs.png")

    # backgrounds for per-prediction explanations in the app
    np.savez(ROOT / "models" / "shap_background.npz", crop_background=shap.kmeans(X[tr], 10).data,
             fert_background=fert_bg)

    summary = {
        "crop": {"explained_rows": int(len(sample)), "background": "kmeans(20) of 80% train split",
                 "nsamples": 256,
                 "mean_abs_shap_by_feature": dict(sorted(
                     {f: float(v) for f, v in zip(NUM_COLS_CROP, crop_mean_abs.mean(axis=1))}.items(),
                     key=lambda kv: -kv[1]))},
        "fertilizer": {"explained_rows": int(len(fdf)),
                       "explainer": "KernelExplainer, background = 20 random rows (seed 42), nsamples 256 (exact for 8 features)",
                       "mean_abs_shap_by_feature": dict(sorted(
                           {f: float(v) for f, v in zip(groups, fert_mean_abs.mean(axis=1))}.items(),
                           key=lambda kv: -kv[1]))},
    }
    (OUT / "shap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
