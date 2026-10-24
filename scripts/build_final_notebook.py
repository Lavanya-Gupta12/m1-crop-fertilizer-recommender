"""Build notebooks/M1_final.ipynb (unexecuted). Execute/export with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/M1_final.ipynb
    python scripts/build_final_notebook.py --strip-startup-logs
    jupyter nbconvert --to html notebooks/M1_final.ipynb
    python scripts/build_final_notebook.py --wrap-html
Needs models/ (see the "Reproduce" section) and results/ from the logged runs.
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "M1_final.ipynb"

# TensorFlow start-up lines printed to stderr before logging is configured. Only stderr outputs
# made up *entirely* of these lines are removed; anything else (warnings, errors) is kept.
STARTUP_LOG = re.compile(r"^(WARNING: All log messages before absl::InitializeLog\(\).*|"
                         r"I0000 .*port\.cc.*oneDNN custom operations are on.*|\s*)$")

if "--wrap-html" in sys.argv:
    # wrap long code lines so nothing is cut off on screen or when printed to PDF
    html_path = NB_PATH.with_suffix(".html")
    css = ("<style>pre, .jp-InputArea pre, .jp-OutputArea pre, .highlight pre "
           "{white-space: pre-wrap !important; overflow-wrap: anywhere !important;}</style>")
    html = html_path.read_text(encoding="utf-8")
    if css not in html:
        html_path.write_text(html.replace("</head>", css + "\n</head>", 1), encoding="utf-8")
    print("wrapped code lines in", html_path.name)
    sys.exit(0)

if "--strip-startup-logs" in sys.argv:
    nb = nbf.read(NB_PATH, as_version=4)
    removed = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        keep = [o for o in c.get("outputs", [])
                if not (o.get("name") == "stderr" and all(STARTUP_LOG.match(l) for l in o["text"].splitlines()))]
        removed += len(c["outputs"]) - len(keep)
        c["outputs"] = keep
    nbf.write(nb, NB_PATH)
    print(f"removed {removed} TensorFlow start-up log output(s)")
    sys.exit(0)
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# M1 — Crop & Fertilizer Recommender (final)

Smart Crop Advisory System (SIH25010), module M1.

**Pipeline**
- **Stage 1** — soil/climate → ranked crops with calibrated probabilities.
- **Stage 2** — soil/climate + predicted crop → fertilizer, with a stated source tier.

**Ground rules**
- Every number shown here is logged in `RESULTS.md` with the command that produced it.
- Every metric is reported with a baseline and its fold-to-fold spread (std, ddof = 1).
- Known limitations are stated, not hidden.

**What this notebook does**
- Loads the logged results and the saved models.
- Recomputes the key checks live: test metrics from the saved models, tier coverage, the
  dataset range mismatch, and smoke tests.
- Retraining everything takes about 15 minutes, so it is not repeated here. The commands are in
  the last section.
""")

code("""
import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")           # silence TensorFlow C++ start-up logging
warnings.filterwarnings("ignore", message="IProgress not found")
import json, sys, platform
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import keras, sklearn, shap
R = lambda p: json.loads((ROOT / "results" / p).read_text())
pd.set_option("display.precision", 4); pd.set_option("display.max_colwidth", None)
print(f"Python {platform.python_version()} | keras {keras.__version__} (backend {keras.backend.backend()}) | "
      f"scikit-learn {sklearn.__version__} | shap {shap.__version__}")
""")

md("## 1. Data")
code("""
from task1_calibration import load_xy, NUM_COLS_CROP, SEED
from train_stage2_notebook import load_fert_df
crop_df, X, y, enc = load_xy()
fert_df = load_fert_df()
print(f"Crop data: {crop_df.shape[0]} rows, {len(enc.classes_)} crops, "
      f"{crop_df['label'].value_counts().min()}-{crop_df['label'].value_counts().max()} rows per crop")
print(f"Fertilizer data: {fert_df.shape[0]} rows, {fert_df['label'].nunique()} crop labels, "
      f"{fert_df['soil_type'].nunique()} soil types, {fert_df['fertilizer'].nunique()} fertilizer classes")
display(fert_df["fertilizer"].value_counts().rename("rows").to_frame().T)
shared = sorted(set(crop_df["label"]) & set(fert_df["label"]))
print("Crop names shared by both datasets (after paddy -> rice):", shared)
""")

md("""
## 2. Stage 1 — crop model

Architecture: 7 standardised inputs → Dense 64 ReLU → Dropout 0.2 → Dense 32 ReLU → Softmax 22
(3,318 parameters).

Deployed model: a **5-seed ensemble with temperature scaling**, chosen in Task 1.
`docs/architecture.md` marks each hyperparameter as default, reasoned or validated.
""")
code("""
from task1_calibration import build_crop_model
m = build_crop_model(7, 22); print("parameters:", m.count_params())
""")

md("""
### 2.1 Task 1: calibration

Setup:
- 5-fold stratified CV with the scaler fit inside each fold, 40 epochs.
- 15% of each training fold is held out to fit the temperature T. The test folds are never used
  for fitting or selection.

Finding: the baseline was slightly **under**-confident on real inputs, so T < 1 in every fold.
The real problem was out-of-range inputs.
""")
code("""
b, v = R("task1/baseline.json"), R("task1/variants.json")
lk = b["leakage_check"]
rows = [("leaky-25ep (scaler on all rows)", lk["leaky_scaler_on_all_rows"]["summary"]),
        ("in-fold-25ep", lk["scaler_inside_fold"]["summary"])]
display(pd.DataFrame({n: {f"{k} mean": s[k]["mean"] for k in ("accuracy", "macro_f1")} |
                         {f"{k} std": s[k]["std"] for k in ("accuracy", "macro_f1")} for n, s in rows}).T)
tab = {"class-prior reference": b["prior_reference_summary"]} | {n: r["summary"] for n, r in v["variants"].items()}
cols = ["accuracy", "macro_f1", "ece", "nll", "brier", "mean_confidence"]
display(pd.DataFrame({n: {c: f'{s[c]["mean"]:.4f} ± {s[c]["std"]:.4f}' for c in cols} for n, s in tab.items()}).T)
print("Temperatures per fold (E_ensemble5+TS):", np.round(v["variants"]["E_ensemble5+TS"]["temperatures"], 3))
""")
code("""
display(Image(filename=str(ROOT / "results/task1/baseline_reliability.png"), width=900))
display(Image(filename=str(ROOT / "results/task1/variants_reliability.png"), width=1000))
""")

md("""
### 2.2 Out-of-range inputs

None of the calibration methods fixes out-of-range inputs; some barely change. That is why an
**input-range guard** was added: any feature outside the training min/max ± 10% of its range
sets the status to `out_of_range`.
""")
code("""
pr = pd.DataFrame({n: r["probe_top1_median"] for n, r in v["variants"].items()})
display(pr.round(3))
""")

md("""
### 2.3 Deployed model on the held-out test split (recomputed live from `models/`)

- **Split:** the same 80/20 split as the original notebook (440 test rows).
- **Uncertain flag:** t = 0.75, chosen as the smallest threshold whose accepted predictions
  reach ≥ 99% accuracy on out-of-fold predictions.
""")
code("""
from sklearn.model_selection import train_test_split
from m1.calibration import all_metrics
from m1.recommender import get_recommender
rec = get_recommender()
_, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
p = rec.crop_probs(X[te])
live = all_metrics(p, y[te]); fin = R("task1/final.json")
conf = p.max(1); t = rec.uncertain_threshold
display(pd.DataFrame({"deployed (live)": live, "logged final.json": fin["test_metrics"],
                      "single model A, same split (logged)": fin["test_metrics_A_same_split"]}).T)
print(f"threshold t={t}: coverage {np.mean(conf >= t):.4f}, accuracy on accepted "
      f"{np.mean((p.argmax(1) == y[te])[conf >= t]):.4f}, flagged {int(np.sum(conf < t))}/{len(te)}")
display(pd.DataFrame(fin["oof_coverage_curve"]).set_index("threshold").round(4))
""")

md("""
## 3. Stage 2 — fertilizer, with source tiers (Task 2)

Only 3 crop names overlap between the two datasets. Each answer therefore states its source:

| Tier | Source |
|---|---|
| Tier 1 | The ML model, for crops present in the fertilizer data. |
| Tier 2 | The ML model queried with the crop group `pulses`, for the seven pulse crops. This is an assumption, documented in `docs/fertilizer_tiers.md`. |
| None | "No verified fertilizer data". |
""")
code("""
from m1 import tiers
cov = tiers.coverage(enc.classes_)
display(pd.DataFrame({k: {"crops": ", ".join(c), "count": len(c), "share of 22": f"{len(c)/22:.1%}"}
                      for k, c in cov.items()}).T)
""")

md("""
### 3.1 Range mismatch between the datasets (recomputed live)

The fertilizer data covers a much narrower range of conditions than the crop data. Outside that
range the stage-2 model is extrapolating, and the app warns about it on every answer.
""")
code("""
g = rec.fert_guard
cols5 = ["temperature", "humidity", "N", "K", "P"]
idx = [g.feature_names.index(c) for c in cols5]
served = crop_df[crop_df["label"].isin(cov["tier1"] + cov["tier2"])][cols5].values
inside = (served >= g.lower_[idx]) & (served <= g.upper_[idx])
display(pd.DataFrame({"fertilizer-data min": g.train_min_[idx], "fertilizer-data max": g.train_max_[idx],
                      "crop-data min (Tier 1/2)": served.min(0), "crop-data max (Tier 1/2)": served.max(0),
                      "share of crop rows inside": inside.mean(0)}, index=cols5).round(2))
print(f"Crop-data rows (Tier 1/2 crops) inside the stage-2 range on all 5 features: {inside.all(1).mean():.1%}")
""")

md("""
## 4. Benchmark (Tasks 3 + 4)

Protocol:
- Stage 1: the same stratified 5-fold.
- Stage 2: repeated stratified 5-fold × 3 (n = 99).
- Preprocessing is fit inside each fold, seeds are fixed, and there is no hyperparameter search.
""")
code("""
bm = R("benchmark/benchmark.json")
def bench(stage):
    cols = ["accuracy", "macro_f1", "ece", "nll", "params", "train_s"]
    return pd.DataFrame({n: {c: (f'{r["summary"][c]["mean"]:,.0f}' if c == "params" else
                                 f'{r["summary"][c]["mean"]:.4f} ± {r["summary"][c]["std"]:.4f}') for c in cols}
                         for n, r in bm[stage].items()}).T
display(Markdown("**Stage 1**")); display(bench("stage1"))
display(Markdown("**Stage 2**")); display(bench("stage2"))
def paired(stage, a, b, k):
    d = np.array([f[k] for f in bm[stage][a]["per_fold"]]) - np.array([f[k] for f in bm[stage][b]["per_fold"]])
    return f"{d.mean():+.4f} ± {d.std(ddof=1):.4f} (ahead/tied/behind: {(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()})"
print("Stage 1, RF − MLP accuracy:      ", paired("stage1", "RandomForest (300 trees)", "MLP 64-32 (current)", "accuracy"))
print("Stage 1, LightGBM − MLP accuracy:", paired("stage1", "LightGBM (defaults)", "MLP 64-32 (current)", "accuracy"))
print("Stage 2, RF − emb. NN accuracy:  ", paired("stage2", "RandomForest (one-hot, 300 trees)", "Embedding NN (current)", "accuracy"))
print("Stage 2, RF − emb. NN macro-F1:  ", paired("stage2", "RandomForest (one-hot, 300 trees)", "Embedding NN (current)", "macro_f1"))
print("Stage 2, emb. − one-hot NN NLL:  ", paired("stage2", "Embedding NN (current)", "One-hot NN", "nll"))
""")

md("""
**Conclusions**

Stage 1: **tree models beat the MLP in all 5 folds.**
- RandomForest and LightGBM are ahead by 1.1–1.5 points of accuracy, several times the fold std.
- LightGBM is also better calibrated than the deployed ensemble + TS.

Stage 2: **RandomForest vs the embedding NN is within noise.**
- The paired differences are smaller than their split-to-split std.
- The test folds are ~20 rows, and the 15 splits are not independent.
- RF was ahead or tied in 14 of 15 splits on accuracy, so the direction favours RF but the size
  of the gap can't be measured with 99 rows.
- The NN has the better NLL in all 15 splits.
- Embeddings beat one-hot inputs on NLL in all 15 splits.

**Decision: neural networks for both stages,** because the project scope is deep learning.
- Stage 1 keeps the MLP ensemble + TS. Stage 2 uses the embedding NN, retrained leak-free on all
  99 rows.
- The cost is about 1.3 points of stage-1 accuracy, plus approximate (Kernel) rather than exact
  (Tree) SHAP.
- The tree models are kept as benchmark results.
""")

md("## 5. Explainability (SHAP on the deployed models)")
code("""
s = R("shap/shap_summary.json")
display(pd.DataFrame({"crop model": s["crop"]["mean_abs_shap_by_feature"],
                      "fertilizer model": s["fertilizer"]["mean_abs_shap_by_feature"]}).round(4))
display(Image(filename=str(ROOT / "results/shap/crop_shap_mean_abs.png"), width=850))
display(Image(filename=str(ROOT / "results/shap/fert_shap_mean_abs.png"), width=850))
""")
md("""
**Reading the plots**
- **Crop:** humidity, rainfall and N/P/K drive the crop choice; pH matters least.
- **Fertilizer:** the choice is driven by N/P/K levels. Crop and soil type contribute little,
  which fits fertilizer names that are NPK grades.
- **Per prediction:** the app shows the top-3 SHAP drivers for every answer.
""")

md("""
## 6. End-to-end examples (`recommend()`, the function the Gradio app calls)

Inputs: the rice-like smoke test, class means from the data, and out-of-range cases.
""")
code("""
from m1.recommender import recommend
cm = lambda c: dict(zip(["N","P","K","temperature","humidity","ph","rainfall"],
                        crop_df.loc[crop_df["label"] == c, NUM_COLS_CROP].mean().round(1)))
cases = {"rice-like (N90 P42 K43 20.8C 82% pH6.5 202mm), Clayey": dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=6.5, rainfall=202, soil_type="Clayey", moisture=45),
         "chickpea class mean, Loamy": cm("chickpea") | dict(soil_type="Loamy", moisture=40),
         "mango class mean, Loamy": cm("mango") | dict(soil_type="Loamy", moisture=40),
         "rainfall 3000 mm, rest rice-like": dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=6.5, rainfall=3000, soil_type="Clayey", moisture=45),
         "pH 14, rest rice-like": dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=14, rainfall=202, soil_type="Clayey", moisture=45)}
rows = {}
for name, kw in cases.items():
    o = recommend(**kw); f = o["fertilizer"]
    rows[name] = {"status": o["status"], "top-3 crops": ", ".join(f"{t['crop']} {t['probability']:.1%}" for t in o["top3"]),
                  "fertilizer": f["fertilizer"] or "—", "fert. conf (uncalibrated)": "—" if f["confidence"] is None else f"{f['confidence']:.2f}",
                  "tier": f["tier"] or "none", "stage-2 range warnings": len(f["warnings"]),
                  "why (crop)": o["reasons"]["crop"].split(": ", 1)[1]}
display(pd.DataFrame(rows).T)
""")

md("""
## 7. Known limitations

1. **The datasets barely overlap.**
   - No row has both stage-1 inputs and a fertilizer label, so end-to-end accuracy can't be
     measured. Only per-stage metrics and coverage are reported.
   - Only 1.7% of realistic stage-1 inputs fall inside the fertilizer data's range, so stage 2
     usually extrapolates.
   - When it extrapolates, the NN can be highly confident (100% on the chickpea example). The
     range warning is the safeguard.
2. **Coverage.** Fertilizer answers cover 10 of 22 crops (3 Tier 1, 7 Tier 2). Tier 2 is an
   assumption that can't be validated: there are no per-pulse labels.
3. **Stage 2 is tiny.** 99 rows, ~20-row test folds and wide spreads (macro-F1 ± 0.08). Its
   confidences are uncalibrated (under-confident in CV, mean confidence 0.80 vs accuracy 0.93).
4. **Trees beat the stage-1 NN** in every fold. The NN is kept by project-scope decision.
5. **No hyperparameter search** was run, and the 40-epoch policy is a default. Calibration, the
   ensemble and the embeddings-vs-one-hot choice are the only validated choices.
6. **No dosage output** (the dataset has fertilizer names only), and no soil type in stage 1 (the
   crop data has no soil column).
7. **Gradio share links are temporary** (≤ 1 week). The team hub needs permanent hosting.
""")

md("""
## 8. Reproduce

Environment: WSL2 Ubuntu 24.04, Python 3.12.3, TensorFlow 2.21 CPU, plus `requirements.txt`.
LightGBM also needs `sudo apt install -y libgomp1`.

```
python scripts/task1_calibration.py baseline
python scripts/task1_calibration.py variants
python scripts/task1_calibration.py final --variant "E_ensemble5+TS"
python scripts/benchmark.py
python scripts/train_stage2.py
python scripts/shap_summary.py
python scripts/smoke_test.py
python scripts/build_final_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/M1_final.ipynb
python scripts/build_final_notebook.py --strip-startup-logs && jupyter nbconvert --to html notebooks/M1_final.ipynb
python scripts/build_final_notebook.py --wrap-html   # then print the HTML to PDF (e.g. Edge/Chrome headless --print-to-pdf)
python app.py --share
```
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NB_PATH.parent.mkdir(exist_ok=True)
nbf.write(nb, NB_PATH)
print("wrote", NB_PATH)
