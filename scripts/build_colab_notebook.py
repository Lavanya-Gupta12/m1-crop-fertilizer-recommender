"""Build notebooks/M1_colab.ipynb: a self-contained, presentation-ready Colab notebook.

All model code is written inline in the notebook (no imports from this repo), so it can be shown
and explained cell by cell. QUICK_MODE=True runs in a few minutes on a CPU; QUICK_MODE=False
reproduces the numbers in RESULTS.md (TensorFlow 2.21 CPU; other TF versions may differ slightly).

Usage: python scripts/build_colab_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


# =============================================================================================
md("""
# M1 — Crop & Fertilizer Recommender

Smart Crop Advisory System (SIH25010), module M1.

**What this notebook does**
- **Stage 1** reads soil and climate values and suggests the **top-3 crops**, with probabilities
  that are honest (calibrated).
- **Stage 2** suggests a **fertilizer** for the top crop, and says where the answer came from
  (its *tier*).

**How to run it:** *Runtime → Run all*, or run one cell at a time with Shift+Enter.

**Speed:**
- `QUICK_MODE = True` runs in about 5 minutes on a free Colab CPU. The crop network gets fewer
  epochs and seeds, so its numbers are lower and the rice example may show "uncertain".
- `QUICK_MODE = False` is the full run (about 10 minutes on a fast PC, 15–20 on Colab). It reproduces the numbers in the
  project's `RESULTS.md`, which were measured with TensorFlow 2.21 on CPU; other versions can
  differ slightly.
""")

md("""
## 1. Setup

These settings control everything below:
- **`SEED`** fixes all random numbers, so every run gives the same result.
- **`QUICK_MODE`** switches between the fast demo and the full run.
""")
code("""
QUICK_MODE = True   # True: fast demo (~5 min on CPU). False: full run, reproduces RESULTS.md

SEED = 42
N_FOLDS = 5          # 5-fold cross-validation
BATCH = 16
CAL_FRACTION = 0.15  # share of each training fold kept aside to calibrate probabilities
if QUICK_MODE:
    EPOCHS, N_SEEDS, STAGE2_REPEATS, SHAP_ROWS_PER_CLASS, SHAP_FERT_ROWS = 10, 2, 1, 2, 30
else:
    EPOCHS, N_SEEDS, STAGE2_REPEATS, SHAP_ROWS_PER_CLASS, SHAP_FERT_ROWS = 40, 5, 3, 10, 99
FERT_EPOCHS = 40     # the fertilizer net trains in seconds (99 rows), so it always gets the full 40 epochs
print(f"QUICK_MODE={QUICK_MODE}: {EPOCHS} epochs, {N_SEEDS} seeds per ensemble, "
      f"{STAGE2_REPEATS} repeat(s) of stage-2 CV")
""")

md("""
Install anything Colab is missing (only packages that aren't already there), then import the
libraries.

The seed function also switches TensorFlow into **deterministic mode**, so a rerun gives exactly
the same numbers.
""")
code("""
import importlib.util, subprocess, sys
for pkg in ["kagglehub", "lightgbm", "shap", "gradio"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, time, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide TensorFlow start-up messages
warnings.filterwarnings("ignore")                    # keep the presentation output readable
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
import tensorflow as tf, keras
tf.get_logger().setLevel("ERROR")                    # no retracing warnings
from keras import layers
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import shap, gradio as gr

def set_seeds(seed):
    keras.utils.set_random_seed(seed)                 # Python, NumPy and TensorFlow seeds
    tf.config.experimental.enable_op_determinism()    # identical results on every run

pd.set_option("display.precision", 4); pd.set_option("display.max_colwidth", None)
T_START = time.time()
print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | shap {shap.__version__} | gradio {gr.__version__}")
""")

# =============================================================================================
md("""
## 2. Data

Two public Kaggle datasets are downloaded with `kagglehub` (no login needed):
- **Crop data:** 2,200 rows, 22 crops, 100 rows each. Features are N, P, K, temperature,
  humidity, pH and rainfall.
- **Fertilizer data:** 99 rows. Features are temperature, humidity, moisture, N, K, P, soil type
  and crop, and the label is one of 7 fertilizers.

Crop names are cleaned the same way in both datasets, and `paddy` is renamed to `rice`.
""")
code("""
import kagglehub
crop_csv = Path(kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")) / "Crop_recommendation.csv"
fert_csv = Path(kagglehub.dataset_download("gdabhishek/fertilizer-prediction")) / "Fertilizer Prediction.csv"
clean = lambda s: str(s).strip().lower().replace(" ", "")

crop_df = pd.read_csv(crop_csv).dropna().drop_duplicates()
crop_df["label"] = crop_df["label"].map(clean)
fert_df = (pd.read_csv(fert_csv).dropna().drop_duplicates()
           .rename(columns={"Temparature": "temperature", "Humidity ": "humidity", "Soil Type": "soil_type",
                            "Crop Type": "label", "Nitrogen": "N", "Potassium": "K", "Phosphorous": "P",
                            "Fertilizer Name": "fertilizer"})
           .reset_index(drop=True))
fert_df["label"] = fert_df["label"].map(clean).replace({"paddy": "rice"})

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
crop_enc = LabelEncoder().fit(sorted(crop_df["label"].unique()))
X = crop_df[FEATURES].values.astype(float)
y = crop_enc.transform(crop_df["label"])
CROPS, N_CLASSES = crop_enc.classes_, len(crop_enc.classes_)

print(f"Crop data: {X.shape[0]} rows, {N_CLASSES} crops")
print(f"Fertilizer data: {len(fert_df)} rows, {fert_df['label'].nunique()} crops, "
      f"{fert_df['soil_type'].nunique()} soil types, {fert_df['fertilizer'].nunique()} fertilizers")
print("Crop names in both datasets:", sorted(set(crop_df["label"]) & set(fert_df["label"])))
display(crop_df.head(3))
""")

# =============================================================================================
md("""
## 3. Crop model: train and cross-validate

**The network** is a small feed-forward net:
- **Layers:** 7 inputs → 64 → 32 → 22 outputs (one per crop), about 3,300 parameters, which is
  small for 2,200 rows.
- **Dropout (0.2):** switches off random neurons while training, so the network can't simply
  memorise the data.
- **Softmax:** turns the 22 outputs into probabilities that sum to 1.
- **Logits layer:** we keep the scores just before the softmax separate, because Section 4 needs
  them.
""")
code("""
def build_crop_model(n_features=7, n_classes=22):
    inputs = layers.Input(shape=(n_features,), name="numeric_input")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    logits = layers.Dense(n_classes, name="logits")(x)             # raw scores
    probs = layers.Activation("softmax", name="crop_output")(logits)
    model = keras.Model(inputs, probs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_crop_model(X_train, y_train, seed):
    set_seeds(seed)
    model = build_crop_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    return model

def get_logits(model, X_in):
    logit_model = keras.Model(model.inputs, model.get_layer("logits").output)
    return np.asarray(logit_model.predict(X_in, verbose=0), dtype=np.float64)

build_crop_model().summary()
""")

md("""
**How we score it.** Every number is reported with a baseline and its spread across folds:
- **Accuracy and macro-F1:** how often the top crop is right (macro-F1 weights every crop
  equally).
- **ECE (expected calibration error):** when the model says 90%, is it right about 90% of the
  time? Lower is better.
- **NLL and Brier:** probability-quality scores. Lower is better.
""")
code("""
def ece(probs, y_true, n_bins=15):
    conf = probs.max(axis=1); correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        if in_bin.any():
            total += in_bin.sum() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return total / len(y_true)

def metrics(probs, y_true):
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1)
    onehot = np.eye(probs.shape[1])[y_true]
    return {"accuracy": accuracy_score(y_true, probs.argmax(1)),
            "macro_f1": f1_score(y_true, probs.argmax(1), average="macro"),
            "ece": ece(probs, y_true), "nll": -np.mean(np.log(p_true)),
            "brier": np.mean(np.sum((probs - onehot) ** 2, axis=1))}

def mean_std(rows):   # mean ± std over folds (std with ddof=1)
    df = pd.DataFrame(rows)
    return df.mean().map("{:.4f}".format) + " ± " + df.std(ddof=1).map("{:.4f}".format)
""")

md("""
**5-fold cross-validation, without leakage**
- **Folds:** the data is split into 5 parts. Each part is tested once, by a model trained on the
  other 4.
- **No leakage:** the `StandardScaler` (which puts features on the same scale) is fit **only on
  each fold's training rows**, so test rows never influence training.
- **Calibration split:** 15% of each training fold is kept aside, untouched, for Section 4.
- **Seeds:** each fold trains `N_SEEDS` networks with different seeds, used for the ensemble in
  Section 4.
""")
code("""
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = []
for fold, (tr, te) in enumerate(skf.split(X, y)):
    fit_idx, cal_idx = train_test_split(tr, test_size=CAL_FRACTION, stratify=y[tr], random_state=SEED + fold)
    folds.append((fit_idx, cal_idx, te))

test_logits = [np.zeros((len(y), N_CLASSES)) for _ in range(N_SEEDS)]   # out-of-fold scores, per seed
cal_logits = []                                                         # per fold: list over seeds
t0 = time.time()
for fold, (fit_idx, cal_idx, te) in enumerate(folds):
    scaler = StandardScaler().fit(X[fit_idx])          # fit on training rows only
    cal_logits.append([])
    for s in range(N_SEEDS):
        model = train_crop_model(scaler.transform(X[fit_idx]), y[fit_idx], seed=SEED + fold + 100 * s)
        test_logits[s][te] = get_logits(model, scaler.transform(X[te]))
        cal_logits[fold].append(get_logits(model, scaler.transform(X[cal_idx])))
    print(f"fold {fold + 1}/{N_FOLDS} done ({time.time() - t0:.0f}s)")

single = [metrics(softmax(test_logits[0][te], axis=1), y[te]) for _, _, te in folds]
prior = [metrics(np.tile(np.bincount(y[f], minlength=N_CLASSES) / len(f), (len(te), 1)), y[te]) for f, _, te in folds]
display(pd.DataFrame({"single network (seed 42)": mean_std(single), "baseline: guess by class share": mean_std(prior)}).T)
""")

# =============================================================================================
md("""
## 4. Calibration: making the probabilities honest

A model can be right but still report the wrong confidence, so we fix this in two steps:
- **Temperature scaling:** divide the scores by one number **T**, then apply softmax. T > 1
  makes the model less sure, and T < 1 more sure. T is learned **only on the calibration split**,
  never on test data.
- **5-seed ensemble:** average the probabilities of networks trained with different seeds. This
  is usually more accurate and better calibrated.
""")
code("""
def fit_temperature(logits, y_true):
    def nll_at(log_t):
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return -np.mean(lp[np.arange(len(y_true)), y_true])
    res = minimize_scalar(nll_at, bounds=np.log([0.05, 20.0]), method="bounded")
    return float(np.exp(res.x))

def ensemble_logits(logit_list):   # average the probabilities, return log-probabilities
    return np.log(np.clip(np.mean([softmax(l, axis=1) for l in logit_list], axis=0), 1e-12, 1.0))

variants = {"single": [], "single + temperature": [], f"ensemble ({N_SEEDS} seeds)": [], f"ensemble + temperature": []}
oof = {name: np.zeros((len(y), N_CLASSES)) for name in variants}
temps = []
for fold, (fit_idx, cal_idx, te) in enumerate(folds):
    single_te, single_cal = test_logits[0][te], cal_logits[fold][0]
    ens_te = ensemble_logits([test_logits[s][te] for s in range(N_SEEDS)])
    ens_cal = ensemble_logits(cal_logits[fold])
    t_single, t_ens = fit_temperature(single_cal, y[cal_idx]), fit_temperature(ens_cal, y[cal_idx])
    temps.append(t_ens)
    for name, z in zip(variants, [single_te, single_te / t_single, ens_te, ens_te / t_ens]):
        p = softmax(z, axis=1); oof[name][te] = p; variants[name].append(metrics(p, y[te]))

display(pd.DataFrame({k: mean_std(v) for k, v in variants.items()}).T)
print("Temperature T per fold (ensemble):", np.round(temps, 3), "-> T < 1 means the raw model was UNDER-confident")
""")

md("""
**Reliability diagram:** each bar is a group of predictions with similar confidence.
- **Perfect calibration:** bars touch the diagonal.
- **Below the diagonal:** over-confident.
- **Above the diagonal:** under-confident.
""")
code("""
def reliability(ax, probs, y_true, title, n_bins=15):
    conf = probs.max(1); correct = probs.argmax(1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    accs = [correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)]
    ax.bar(centers, accs, width=1 / n_bins, edgecolor="black", alpha=0.7, label="accuracy in bin")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="accuracy", title=f"{title}\\nECE = {ece(probs, y_true):.4f}")
    ax.legend(loc="upper left", fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
reliability(axes[0], oof["single"], y, "Single network (raw)")
reliability(axes[1], oof["ensemble + temperature"], y, "Ensemble + temperature (deployed)")
plt.tight_layout(); plt.show()
""")

md("""
**The deployed crop model**
- **Training:** an ensemble trained on 80% of the data, with T fit on a held-out part of that
  80%. It's then tested once on the untouched 20%.
- **Two safety checks:**
  1. **Input-range guard:** if any input is outside the training range (± 10%), the answer is
     flagged *out of range*.
  2. **"Uncertain" flag:** if the top crop's probability is below a threshold, the answer is
     flagged *uncertain*. The threshold is the smallest one where the accepted predictions are
     ≥ 99% correct (from the cross-validation above).
""")
code("""
tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
fit_idx, cal_idx = train_test_split(tr_idx, test_size=CAL_FRACTION, stratify=y[tr_idx], random_state=SEED)
crop_scaler = StandardScaler().fit(X[fit_idx])
crop_models = [train_crop_model(crop_scaler.transform(X[fit_idx]), y[fit_idx], seed=SEED + 100 * k) for k in range(N_SEEDS)]
T = fit_temperature(ensemble_logits([get_logits(m, crop_scaler.transform(X[cal_idx])) for m in crop_models]), y[cal_idx])
logit_models = [keras.Model(m.inputs, m.get_layer("logits").output) for m in crop_models]

def crop_probs(X_raw):
    # calibrated probabilities of the deployed ensemble, for raw (unscaled) input rows
    x = crop_scaler.transform(np.atleast_2d(np.asarray(X_raw, dtype=float)))
    logits = [np.asarray(keras.ops.convert_to_numpy(m([x], training=False)), dtype=np.float64) for m in logit_models]
    return softmax(ensemble_logits(logits) / T, axis=1)

# input-range guard: training min/max, widened by 10% of each feature's range
lo, hi = X[tr_idx].min(0), X[tr_idx].max(0)
CROP_LOW, CROP_HIGH = lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)

# "uncertain" threshold from the out-of-fold predictions of the ensemble + temperature
conf_oof, ok_oof = oof["ensemble + temperature"].max(1), oof["ensemble + temperature"].argmax(1) == y
grid = np.round(np.arange(0.30, 1.0, 0.05), 2)
good = [t for t in grid if ok_oof[conf_oof >= t].mean() >= 0.99]
THRESHOLD = min(good) if good else 0.75

p_test = crop_probs(X[te_idx])
print(f"T = {T:.4f} | uncertain threshold = {THRESHOLD}")
display(pd.DataFrame({"deployed ensemble + T, held-out 20% test": metrics(p_test, y[te_idx])}).T)
""")

# =============================================================================================
md("""
## 5. Fertilizer model and tiers

**The network**
- **Inputs:** 6 numbers (temperature, humidity, moisture, N, K, P) plus two categories (soil type
  and crop).
- **Embeddings:** each category is turned into a short learned vector: soil 5 → 2 numbers,
  crop 11 → 5 numbers. Similar soils or crops can end up with similar vectors.
- **Rest of the network:** 64 → 32 → 7 fertilizers, the same pattern as the crop network.
- **No leakage:** the scaler and encoders are fit inside `fit()`, on the training rows only.
""")
code("""
NUM_FERT = ["temperature", "humidity", "Moisture", "N", "K", "P"]
FERT_CLASSES = sorted(fert_df["fertilizer"].unique())
y_fert = np.array([FERT_CLASSES.index(v) for v in fert_df["fertilizer"]])
SOILS, FERT_CROPS = sorted(fert_df["soil_type"].unique()), sorted(fert_df["label"].unique())

def build_fert_nn(n_soil, n_crop, n_classes, onehot=False):
    num_in = layers.Input(shape=(len(NUM_FERT),), name="numeric_input")
    if onehot:   # alternative used in the benchmark: categories as one-hot vectors
        cat_in = layers.Input(shape=(n_soil + n_crop,), name="onehot_input")
        x = layers.Concatenate()([num_in, cat_in]); inputs = [num_in, cat_in]
    else:        # deployed: learned embeddings
        soil_in, crop_in = layers.Input(shape=(1,), name="soil_input"), layers.Input(shape=(1,), name="crop_input")
        soil_vec = layers.Flatten()(layers.Embedding(n_soil, max(2, n_soil // 2))(soil_in))
        crop_vec = layers.Flatten()(layers.Embedding(n_crop, max(2, n_crop // 2))(crop_in))
        x = layers.Concatenate()([num_in, soil_vec, crop_vec]); inputs = [num_in, soil_in, crop_in]
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="fert_output")(x)
    model = keras.Model(inputs, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class FertNN:
    # fertilizer network with its preprocessing fitted on the training rows only
    def __init__(self, seed, onehot=False):
        self.seed, self.onehot = seed, onehot
        self.soil_enc, self.crop_enc = LabelEncoder().fit(SOILS), LabelEncoder().fit(FERT_CROPS)
    def _inputs(self, df):
        xn = self.scaler.transform(df[NUM_FERT].values)
        s, c = self.soil_enc.transform(df["soil_type"]), self.crop_enc.transform(df["label"])
        if self.onehot:
            return [xn, np.hstack([np.eye(len(SOILS))[s], np.eye(len(FERT_CROPS))[c]])]
        return [xn, s, c]
    def fit(self, df, y_true):
        self.scaler = StandardScaler().fit(df[NUM_FERT].values)
        set_seeds(self.seed)
        self.model = build_fert_nn(len(SOILS), len(FERT_CROPS), len(FERT_CLASSES), self.onehot)
        self.model.fit(self._inputs(df), y_true, epochs=FERT_EPOCHS, batch_size=BATCH, verbose=0)
        return self
    def predict_proba(self, df):
        return np.asarray(self.model.predict(self._inputs(df), verbose=0), dtype=float)
    def predict_proba_fast(self, df):   # direct call: quicker for single rows and SHAP
        return np.asarray(keras.ops.convert_to_numpy(self.model(self._inputs(df), training=False)), dtype=float)

build_fert_nn(len(SOILS), len(FERT_CROPS), len(FERT_CLASSES)).summary()
fert_model = FertNN(seed=SEED).fit(fert_df, y_fert)        # deployed: all 99 rows
f_lo, f_hi = fert_df[NUM_FERT].values.min(0), fert_df[NUM_FERT].values.max(0)
FERT_LOW, FERT_HIGH = f_lo - 0.1 * (f_hi - f_lo), f_hi + 0.1 * (f_hi - f_lo)
""")

md("""
**Tiers: where a fertilizer answer comes from**

Only 3 crop names (rice, maize, cotton) appear in both datasets, so every answer says which tier
produced it:

| Tier | Source |
|---|---|
| **Tier 1** | The crop is in the fertilizer data, so the ML model answers directly. |
| **Tier 2** | The crop is a pulse, so the model is asked using the dataset's single `pulses` group. This is an assumption, stated openly. |
| **No data** | The other crops get "No verified fertilizer data". Nothing is invented. |
""")
code("""
TIER1 = {"rice", "maize", "cotton"}
TIER2 = {c: "pulses" for c in ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"]}

def resolve_tier(crop):
    if crop in TIER1:
        return 1, crop, f"Tier 1: ML model trained on '{crop}' rows."
    if crop in TIER2:
        return 2, TIER2[crop], f"Tier 2: proxy via crop group - '{crop}' answered with the 'pulses' rows (assumption)."
    return None, None, "No verified fertilizer data for this crop."

tiers = {c: resolve_tier(c)[0] for c in CROPS}
coverage = pd.DataFrame({"crops": {t: ", ".join(c for c in CROPS if tiers[c] == k)
                                   for t, k in [("Tier 1", 1), ("Tier 2", 2), ("No data", None)]}})
coverage["count"] = coverage["crops"].str.count(",") + 1
display(coverage)

# how often are real crop-data inputs inside the fertilizer data's range? (5 shared features)
shared = ["temperature", "humidity", "N", "K", "P"]
col = [NUM_FERT.index(c) for c in shared]
served = crop_df[crop_df["label"].map(lambda c: tiers[c] is not None)][shared].values
inside = (served >= FERT_LOW[col]) & (served <= FERT_HIGH[col])
print(f"Tier 1/2 crop rows inside the fertilizer data's range on all 5 shared features: {inside.all(1).mean():.1%}")
print("-> stage 2 usually extrapolates, so the app shows a warning whenever it does.")
""")

# =============================================================================================
md("""
## 6. Benchmark: is the neural network the best choice?

Every model gets the same folds, scaler fit inside each fold, and fixed seeds. We don't assume
the network wins.
- **Stage 1 (crops):** Logistic Regression, Random Forest and LightGBM (gradient-boosted trees)
  against our network.
- **Stage 2 (fertilizer, only 99 rows):** majority-class guess, Random Forest, our embedding
  network and a one-hot network.
""")
code("""
def tree_nodes(m):
    if isinstance(m, RandomForestClassifier):
        return sum(e.tree_.node_count for e in m.estimators_)
    return sum(2 * t["num_leaves"] - 1 for t in m.booster_.dump_model()["tree_info"])

bench1 = {name: [] for name in ["Logistic Regression", "Random Forest", "LightGBM", "Neural network (ours)"]}
for fold, (tr, te) in enumerate(skf.split(X, y)):
    for name, clf in [("Logistic Regression", LogisticRegression(max_iter=5000, random_state=SEED)),
                      ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
                      ("LightGBM", LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=4))]:
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        t0 = time.perf_counter(); pipe.fit(X[tr], y[tr]); secs = time.perf_counter() - t0
        size = clf.coef_.size + clf.intercept_.size if name == "Logistic Regression" else tree_nodes(clf)
        bench1[name].append({**metrics(pipe.predict_proba(X[te]), y[te]), "params / tree nodes": size, "train s": secs})
    scaler = StandardScaler().fit(X[tr])
    t0 = time.perf_counter(); m = train_crop_model(scaler.transform(X[tr]), y[tr], seed=SEED + fold); secs = time.perf_counter() - t0
    p = np.asarray(m.predict(scaler.transform(X[te]), verbose=0), dtype=float)
    bench1["Neural network (ours)"].append({**metrics(p, y[te]), "params / tree nodes": m.count_params(), "train s": secs})

cols = ["accuracy", "macro_f1", "ece", "nll", "params / tree nodes", "train s"]
table1 = pd.DataFrame({k: mean_std([{c: r[c] for c in cols if c != "params / tree nodes"} for r in v]) for k, v in bench1.items()}).T
table1.insert(4, "params / tree nodes", [f'{np.mean([r["params / tree nodes"] for r in v]):,.0f}' for v in bench1.values()])
display(Markdown("**Stage 1: crop recommendation (5-fold CV)**")); display(table1)
diff = np.array([r["accuracy"] for r in bench1["Random Forest"]]) - np.array([r["accuracy"] for r in bench1["Neural network (ours)"]])
print(f"Random Forest - neural network accuracy: {diff.mean():+.4f} ± {diff.std(ddof=1):.4f}, "
      f"RF ahead in {(diff > 0).sum()}/{len(diff)} folds")
""")
code("""
def rf_pipeline(seed):
    pre = ColumnTransformer([("num", StandardScaler(), NUM_FERT),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["soil_type", "label"])])
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))])

bench2 = {name: [] for name in ["Majority class (baseline)", "Random Forest", "Embedding network (ours)", "One-hot network"]}
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=STAGE2_REPEATS, random_state=SEED)
for k, (tr, te) in enumerate(rskf.split(fert_df, y_fert)):
    d_tr, d_te = fert_df.iloc[tr], fert_df.iloc[te]
    majority = np.zeros((len(te), len(FERT_CLASSES))); majority[:, np.bincount(y_fert[tr]).argmax()] = 1
    bench2["Majority class (baseline)"].append(metrics(majority, y_fert[te]))
    bench2["Random Forest"].append(metrics(rf_pipeline(SEED + k).fit(d_tr, y_fert[tr]).predict_proba(d_te), y_fert[te]))
    bench2["Embedding network (ours)"].append(metrics(FertNN(SEED + k).fit(d_tr, y_fert[tr]).predict_proba(d_te), y_fert[te]))
    bench2["One-hot network"].append(metrics(FertNN(SEED + k, onehot=True).fit(d_tr, y_fert[tr]).predict_proba(d_te), y_fert[te]))

table2 = pd.DataFrame({k: mean_std([{c: r[c] for c in ["accuracy", "macro_f1", "ece", "nll"]} for r in v]) for k, v in bench2.items()}).T
display(Markdown(f"**Stage 2: fertilizer (repeated 5-fold × {STAGE2_REPEATS}, test folds of ~20 rows)**")); display(table2)
""")
md("""
**What the benchmark shows (full mode)**
- **Stage 1: tree models beat our network in every fold.** Random Forest gets about 0.995
  accuracy against about 0.98.
- **Stage 2: Random Forest and our embedding network are within noise.** The difference is
  smaller than the fold-to-fold spread, and the test folds are only ~20 rows.
- **Why we still use the network:** the project scope is deep learning. The network's
  probabilities are calibrated (Section 4), and both stages share one design. The tree results
  are reported openly, not hidden.
- **Embeddings vs one-hot:** in stage 2, embeddings beat one-hot inputs on probability quality
  (NLL).

In quick mode the crop network trains for fewer epochs, so its numbers are lower than in full mode.
""")

# =============================================================================================
md("""
## 7. Explainability (SHAP)

SHAP measures how much each input pushed a prediction up or down. A longer bar means that
feature matters more.
- **Crop model:** averaged over test rows and all 22 crops.
- **Fertilizer model:** averaged over its rows and all 7 fertilizers. Soil type and crop each
  count as one feature.
""")
code("""
def stacked_shap_bar(mean_abs, feature_names, class_names, title):
    order = np.argsort(mean_abs.sum(axis=1))
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)[::4]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(feature_names) + 2.5)); left = np.zeros(len(order))
    for c, name in enumerate(class_names):
        ax.barh(np.array(feature_names)[order], mean_abs[order, c], left=left, color=colors[c], label=name)
        left += mean_abs[order, c]
    ax.set(xlabel="mean |SHAP value| (probability), stacked over classes", title=title)
    ax.legend(fontsize=7, ncol=2 if len(class_names) > 10 else 1, loc="lower right"); plt.tight_layout(); plt.show()

rng = np.random.default_rng(SEED)
rows = np.concatenate([rng.choice(te_idx[y[te_idx] == c], SHAP_ROWS_PER_CLASS, replace=False) for c in range(N_CLASSES)])
crop_sv = np.asarray(shap.KernelExplainer(crop_probs, shap.kmeans(X[tr_idx], 20)).shap_values(X[rows], nsamples=256, silent=True))
stacked_shap_bar(np.abs(crop_sv).mean(0), FEATURES, list(CROPS), f"Crop model: mean |SHAP| over {len(rows)} test rows")
""")
code("""
FERT_FEATURES = NUM_FERT + ["soil_type", "crop"]
def fert_encode(df):    # one numeric matrix: 6 numbers, soil id, crop id
    return np.column_stack([df[NUM_FERT].values.astype(float),
                            fert_model.soil_enc.transform(df["soil_type"]), fert_model.crop_enc.transform(df["label"])])
def fert_probs_encoded(Z):
    Z = np.atleast_2d(Z); df = pd.DataFrame(Z[:, :6], columns=NUM_FERT)
    df["soil_type"] = fert_model.soil_enc.classes_[np.rint(Z[:, 6]).astype(int)]
    df["label"] = fert_model.crop_enc.classes_[np.rint(Z[:, 7]).astype(int)]
    return fert_model.predict_proba_fast(df)

Z_fert = fert_encode(fert_df)
fert_bg = Z_fert[rng.choice(len(Z_fert), 20, replace=False)]      # 20 real rows as the reference
fert_sv = np.asarray(shap.KernelExplainer(fert_probs_encoded, fert_bg).shap_values(Z_fert[:SHAP_FERT_ROWS], nsamples=256, silent=True))
stacked_shap_bar(np.abs(fert_sv).mean(0), FERT_FEATURES, FERT_CLASSES, f"Fertilizer model: mean |SHAP| over {SHAP_FERT_ROWS} rows")
""")

# =============================================================================================
md("""
## 8. Demo: `recommend()`

`recommend()` is the one function the app calls. For each input it returns:
- the top-3 crops with calibrated probabilities;
- a status: **ok**, **uncertain** or **out of range**;
- a fertilizer with its tier, plus a warning when the fertilizer model is extrapolating;
- the top SHAP reasons.
""")
code("""
crop_bg = shap.kmeans(X[tr_idx], 10).data

def top_reasons(names, values, label):
    order = np.argsort(-np.abs(values))[:3]
    return f"'{label}': " + ", ".join(f"{names[i]} {values[i]:+.2f}" for i in order)

def range_warnings(names, values, low, high):
    return [f"{n}={v:g} is outside the training range" for n, v, l, h in zip(names, values, low, high) if v < l or v > h]

def recommend(N, P, K, temperature, humidity, ph, rainfall, soil_type=None, moisture=None):
    x = [N, P, K, temperature, humidity, ph, rainfall]
    probs = crop_probs(x)[0]
    order = np.argsort(probs)[::-1][:3]
    top3 = [(CROPS[i], float(probs[i])) for i in order]
    warnings_ = range_warnings(FEATURES, x, CROP_LOW, CROP_HIGH)
    status = "out of range" if warnings_ else ("uncertain" if top3[0][1] < THRESHOLD else "ok")
    k = int(order[0])
    sv = shap.KernelExplainer(lambda Z: crop_probs(Z)[:, k], crop_bg).shap_values(np.atleast_2d(x), nsamples=128, silent=True)
    result = {"top3": top3, "status": status, "warnings": warnings_,
              "why (crop)": top_reasons(FEATURES, np.asarray(sv)[0], top3[0][0]),
              "fertilizer": None, "tier": None, "fertilizer warnings": [], "why (fertilizer)": None}
    tier, fert_crop, provenance = resolve_tier(top3[0][0])
    result["tier"] = provenance
    if tier is not None and soil_type is not None and moisture is not None:
        row = pd.DataFrame([{"temperature": temperature, "humidity": humidity, "Moisture": moisture,
                             "N": N, "K": K, "P": P, "soil_type": soil_type, "label": fert_crop}])
        fp = fert_model.predict_proba_fast(row)[0]; j = int(fp.argmax())
        result["fertilizer"] = f"{FERT_CLASSES[j]} ({fp[j]:.0%}, uncalibrated)"
        result["fertilizer warnings"] = range_warnings(NUM_FERT, [temperature, humidity, moisture, N, K, P], FERT_LOW, FERT_HIGH)
        fsv = shap.KernelExplainer(lambda Z: fert_probs_encoded(Z)[:, j], fert_bg).shap_values(fert_encode(row), nsamples=256, silent=True)
        result["why (fertilizer)"] = top_reasons(FERT_FEATURES, np.asarray(fsv)[0], FERT_CLASSES[j])
    return result

EXAMPLES = {
    "Rice-like": [90, 42, 43, 20.8, 82, 6.5, 202, "Clayey", 45],
    "Pulses (chickpea-like)": [40, 68, 80, 18.9, 17, 7.3, 80, "Loamy", 40],
    "Out-of-range pH (14)": [90, 42, 43, 20.8, 82, 14, 202, "Clayey", 45],
    "Mango (real data row 1100)": [2, 40, 27, 29.73770045, 47.54885174, 5.954626604, 90.09586854, "Loamy", 40],
}
demo_rows = {}
for name, args in EXAMPLES.items():
    r = recommend(*args)
    demo_rows[name] = {"status": r["status"], "top-3 crops": ", ".join(f"{c} {p:.1%}" for c, p in r["top3"]),
                       "fertilizer": r["fertilizer"] or "-", "source": r["tier"],
                       "warnings": len(r["warnings"]) + len(r["fertilizer warnings"]), "why (crop)": r["why (crop)"]}
display(pd.DataFrame(demo_rows).T)
""")

# =============================================================================================
md("""
## 9. Launch the app (Gradio)

The same `recommend()` function, behind a simple web form:
- **Starting state:** the fields are empty. Click an example row to fill them in.
- **Missing inputs:** if a field is left empty, the app asks you to fill it in.
- **`share=True`:** prints a temporary public link (valid for up to a week) that others can open.
""")
code("""
LABELS = ["Nitrogen (N)", "Phosphorous (P)", "Potassium (K)", "Temperature (°C)", "Humidity (%)",
          "pH", "Rainfall (mm)", "Soil Type", "Moisture (%)"]

def app_predict(*values):
    missing = [label for label, v in zip(LABELS, values) if v is None or v == ""]
    if missing:
        return None, "**Please fill all inputs** (or click an example). Missing: " + ", ".join(missing), "", ""
    r = recommend(*values)
    status = {"ok": "**Status:** OK", "uncertain": f"**Status: UNCERTAIN** (top crop below {THRESHOLD:.0%})",
              "out of range": "**Status: INPUT OUT OF RANGE** - the ranking is not reliable"}[r["status"]]
    status += "".join(f"\\n- {w}" for w in r["warnings"])
    fert = f"**{r['fertilizer'] or 'No recommendation'}**  \\n{r['tier']}"
    fert += "".join(f"\\n- fertilizer input {w} (extrapolation)" for w in r["fertilizer warnings"])
    why = f"**Crop:** {r['why (crop)']}" + (f"\\n\\n**Fertilizer:** {r['why (fertilizer)']}" if r["why (fertilizer)"] else "")
    return dict(r["top3"]), status, fert, why

demo = gr.Interface(
    fn=app_predict,
    inputs=[gr.Number(label=l, value=None) for l in LABELS[:7]]
           + [gr.Dropdown(choices=SOILS, label=LABELS[7], value=None), gr.Number(label=LABELS[8], value=None)],
    outputs=[gr.Label(num_top_classes=3, label="Top-3 crops (calibrated)"), gr.Markdown(), gr.Markdown(), gr.Markdown()],
    examples=list(EXAMPLES.values()), example_labels=list(EXAMPLES.keys()), cache_examples=False,
    title="M1 — Crop & Fertilizer Recommender", flagging_mode="never")
demo.launch(share=True)
""")

md("## Key numbers from this run")
code("""
key = {
    "mode": "QUICK" if QUICK_MODE else "FULL",
    "CV single network accuracy": variants["single"],
    "CV ensemble + T accuracy": variants["ensemble + temperature"],
    "CV ECE single -> ensemble + T": f'{np.mean([r["ece"] for r in variants["single"]]):.4f} -> {np.mean([r["ece"] for r in variants["ensemble + temperature"]]):.4f}',
    "CV NLL single -> ensemble + T": f'{np.mean([r["nll"] for r in variants["single"]]):.4f} -> {np.mean([r["nll"] for r in variants["ensemble + temperature"]]):.4f}',
    "deployed test accuracy": f'{metrics(p_test, y[te_idx])["accuracy"]:.4f}',
    "T": f"{T:.4f}", "uncertain threshold": THRESHOLD,
    "stage-2 range overlap": f"{inside.all(1).mean():.1%}",
    "runtime (minutes)": f"{(time.time() - T_START) / 60:.1f}",
}
for k_, v_ in key.items():
    v_ = f'{pd.DataFrame(v_)["accuracy"].mean():.4f}' if isinstance(v_, list) else v_
    print(f"{k_:32s} {v_}")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}
out = ROOT / "notebooks" / "M1_colab.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
