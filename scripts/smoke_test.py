"""Smoke tests for the saved M1 artifacts via the public `recommend()` function.

Usage: python scripts/smoke_test.py
Also checks that the loaded artifacts reproduce the test metrics in results/task1/final.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.model_selection import train_test_split  # noqa: E402

from m1.recommender import get_recommender, recommend  # noqa: E402
from task1_calibration import NUM_COLS_CROP, SEED, load_xy  # noqa: E402


def show(name, out):
    top = ", ".join(f"{t['crop']} {t['probability']:.1%}" for t in out["top3"])
    print(f"\n## {name}\n  status={out['status']}  top3: {top}")
    if out["message"]:
        print(f"  message: {out['message']}")
    for w in out["warnings"]:
        print(f"  warning: {w}")
    f = out["fertilizer"]
    if f:
        print(f"  fertilizer: {f['fertilizer']} conf={f['confidence']} tier={f['tier']} | {f['provenance']}")


def main():
    df, X, y, enc = load_xy()
    rec = get_recommender()

    # integrity: artifacts loaded from disk reproduce the saved test metrics
    _, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    probs = rec.crop_probs(X[te])
    acc = float((probs.argmax(1) == y[te]).mean())
    saved = json.loads((ROOT / "results" / "task1" / "final.json").read_text())["test_metrics"]["accuracy"]
    print(f"reloaded test accuracy={acc:.4f} (final.json: {saved:.4f}) match={abs(acc - saved) < 1e-9}")

    from m1 import tiers
    cov = tiers.coverage(enc.classes_)
    print("tier coverage:", {k: f"{len(v)}/{len(enc.classes_)} {v}" for k, v in cov.items()})

    mango_mean = df.loc[df["label"] == "mango", NUM_COLS_CROP].mean().round(1).tolist()
    chickpea_mean = df.loc[df["label"] == "chickpea", NUM_COLS_CROP].mean().round(1).tolist()
    cases = {
        "rice-like (reference smoke test), Clayey soil, moisture 45":
            dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=6.5, rainfall=202,
                 soil_type="Clayey", moisture=45),
        f"pulses-like (chickpea class mean from the data: {chickpea_mean}), Loamy, moisture 40":
            dict(zip(["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], chickpea_mean),
                 soil_type="Loamy", moisture=40),
        f"no-data crop (mango class mean: {mango_mean}), Loamy, moisture 40":
            dict(zip(["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], mango_mean),
                 soil_type="Loamy", moisture=40),
        "all zeros": dict(N=0, P=0, K=0, temperature=0, humidity=0, ph=0, rainfall=0),
        "pH = 14, otherwise rice-like": dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=14, rainfall=202),
        "rainfall = 3000 mm, otherwise rice-like": dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=6.5, rainfall=3000),
        "humidity = 150 %, temperature = 60 C": dict(N=90, P=42, K=43, temperature=60, humidity=150, ph=6.5, rainfall=202),
    }
    for name, kw in cases.items():
        show(name, recommend(**kw))


if __name__ == "__main__":
    main()
