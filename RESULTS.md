# M1 results log

Every number produced for M1, with the command that produced it.
Conventions: mean ± std over folds, **std uses ddof=1**. ECE = top-label, 15 equal-width bins.
Environment (unless stated): WSL2 Ubuntu 24.04.3, Python 3.12.3, TensorFlow 2.21.0 CPU,
Keras 3.15.1, `KERAS_BACKEND=tensorflow`, seed 42 (+fold index), TF op determinism on.

Command prefix used for every run below (from PowerShell):

```
wsl -d Ubuntu --cd "C:\Users\Lavanya Gupta\Downloads\m1-project" -- bash -lc '. ~/venvs/m1/bin/activate && python scripts/task1_calibration.py baseline'
```

---

## Pre-existing numbers (notebook, Colab, not reproduced bit-for-bit here)

| Label | Value | Notes |
|---|---|---|
| notebook stage-1 5-fold CV | acc 0.9755, macro-F1 0.9755 | Colab TF, 25 epochs, scaler fit on all rows (leaky), no std reported |
| notebook stage-2 test | acc 0.95 (20 rows, 1 error) | single split, no baseline, no CV; not reliable |

---

## Task 1 / step 1 — baseline (no calibration fixes) — 2026-10-25

Command: `python scripts/task1_calibration.py baseline` (outputs: `results/task1/baseline.json`,
`results/task1/baseline_reliability.png`). Data: 2,200 rows, 22 classes, 5-fold stratified CV
(`random_state=42`). Runtime ~130–140 s.

**Reproducibility:** run twice; `baseline.json` (excluding runtime) identical and out-of-fold
logits bit-identical (max abs diff 0.0).

### Stage-1 accuracy: three protocols

| Label | Backend | Epochs | Training data per fold | Scaler | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| **leaky-25ep** | TF 2.21 (WSL) | 25 | 100% of training fold (80% of data) | fit on all 2,200 rows | 0.9732 ± 0.0076 | 0.9728 ± 0.0079 |
| **in-fold-25ep** | TF 2.21 (WSL) | 25 | 100% of training fold (80% of data) | fit on training fold | 0.9727 ± 0.0079 | 0.9724 ± 0.0082 |
| **calibration-baseline-40ep-on-85%-of-fold** | TF 2.21 (WSL) | 40 | 85% of training fold (68% of data); 15% held out for later temperature scaling | fit on the 85% | 0.9786 ± 0.0044 | 0.9785 ± 0.0045 |
| reference: class-prior predictor | – | – | class frequencies of the 85% | – | 0.0455 ± 0.0000 | 0.0040 ± 0.0000 |

- Leakage effect: −0.0005 accuracy (1 of 2,200 rows differs, fold 2). Negligible on this
  dataset, but in-fold scaling is now the standard.
- The leaky-25ep value (0.9732) differs from the notebook's 0.9755 because of backend/RNG
  (Colab TF vs local TF 2.21), not because of a protocol change.
- 40 epochs on 85% scored higher than 25 epochs on 100%. This suggests 25 epochs under-trains;
  Task 3 will settle the epoch policy.

### Stage-1 calibration (calibration-baseline-40ep-on-85%-of-fold, out-of-fold)

| Metric | Per-fold mean ± std | Pooled OOF (2,200 rows) | Class-prior reference |
|---|---|---|---|
| Accuracy | 0.9786 ± 0.0044 | 0.9786 | 0.0455 |
| Macro-F1 | 0.9785 ± 0.0045 | 0.9786 | 0.0040 |
| ECE (15 bins) | 0.0193 ± 0.0021 | 0.0140 | 0.0000 (calibrated but useless) |
| NLL | 0.0663 ± 0.0085 | 0.0663 | 3.0910 (= ln 22) |
| Brier (multiclass, 0–2) | 0.0360 ± 0.0050 | 0.0360 | 0.9545 |
| Mean top-1 confidence | 0.9647 ± 0.0033 | 0.9647 | 0.0455 |

Per-fold ECE is higher than pooled ECE because ECE is biased upward on small samples
(440 rows/fold). Both are reported.

Confidence profile (pooled OOF):
- 70.5% of rows have top-1 ≥ 0.99, and **all of them are correct** (accuracy 1.000). 41.1% have ≥ 0.999.
- 47 errors in total. Median confidence on errors is 0.697; 5 of 47 errors (10.6%) have confidence ≥ 0.9.
- Mean confidence (0.9647) is *below* accuracy (0.9786): on average the model is slightly
  **under**-confident in-distribution, not over-confident.

### Out-of-range probes (top-1 probability across the 5 fold models)

| Probe | min | median | max | Predicted class(es) |
|---|---|---|---|---|
| rice_like (N90 P42 K43 T20.8 H82 pH6.5 R202), in-distribution | 0.598 | 0.728 | 0.772 | rice |
| all zeros | 0.693 | 0.999 | 1.000 | mango, mothbeans |
| all features 2× train max | 1.000 | 1.000 | 1.000 | papaya |
| all features 5× train max | 1.000 | 1.000 | 1.000 | papaya |
| N = 3× max, rest median | 1.000 | 1.000 | 1.000 | cotton |
| P = 3× max, rest median | 0.687 | 1.000 | 1.000 | banana |
| K = 3× max, rest median | 1.000 | 1.000 | 1.000 | grapes |
| temperature = 3× max, rest median | 1.000 | 1.000 | 1.000 | papaya |
| humidity = 3× max, rest median | 0.999 | 1.000 | 1.000 | papaya |
| pH = 3× max, rest median | 0.994 | 1.000 | 1.000 | grapes, mothbeans |
| rainfall = 3× max, rest median | 1.000 | 1.000 | 1.000 | rice |
| pH = 0 (impossible) | 0.529 | 0.947 | 0.979 | grapes, mothbeans, pigeonpeas |
| pH = 14 (impossible) | 0.736 | 1.000 | 1.000 | grapes, mothbeans |
| temperature = 60 °C | 0.998 | 1.000 | 1.000 | papaya |
| humidity = 150% (impossible) | 0.990 | 1.000 | 1.000 | papaya |
| rainfall = 3000 mm | 1.000 | 1.000 | 1.000 | rice |

**Finding:** the "~100% confidence" problem is an **out-of-range** problem. On real
(in-distribution) rows, near-certain predictions are always correct and overall calibration is
already good (ECE 0.014–0.019). On impossible or extreme inputs the model reports 99–100%
confidence for an arbitrary crop.

---

## Task 1 / step 2 — calibration variants A–E — 2026-10-25

Command: `python scripts/task1_calibration.py variants` (same WSL prefix; outputs
`results/task1/variants.json`, `results/task1/variants_reliability.png`). Backend TF 2.21 (WSL),
40 epochs, batch 16. **Training data per fold: 85% of the training fold (68% of all rows).** The
other 15% (calibration split) is used only to fit T and to choose the LS value. Folds, splits
and in-fold scalers are identical to step 1. Runtime 436 s.

- Variant A reproduces the step-1 baseline OOF logits bit-for-bit (`variant_A_reproduces_baseline: true`).
- D's label-smoothing value is chosen by mean calibration-split NLL (never the test folds):
  ls=0.05 → 0.1365, ls=0.1 → 0.1956, so **D uses ls=0.05**.
- E = mean of the softmax outputs of 5 models (seeds 42+fold+100·k, k=0..4; k=0 is A's model).
  Temperature scaling for E is applied to log of that mean.

| Variant | Accuracy | Macro-F1 | ECE | NLL | Brier | Mean conf | Pooled ECE | T per fold |
|---|---|---|---|---|---|---|---|---|
| reference: class prior | 0.0455 ± 0.0000 | 0.0040 ± 0.0000 | 0.0000 | 3.0910 | 0.9545 | 0.0455 | – | – |
| A baseline | 0.9786 ± 0.0044 | 0.9785 ± 0.0045 | 0.0193 ± 0.0021 | 0.0663 ± 0.0085 | 0.0360 ± 0.0050 | 0.9647 | 0.0140 | – |
| B ls=0.05 | 0.9777 ± 0.0041 | 0.9777 ± 0.0042 | 0.0771 ± 0.0044 | 0.1319 ± 0.0090 | 0.0477 ± 0.0037 | 0.9019 | 0.0758 | – |
| B ls=0.1 | 0.9795 ± 0.0036 | 0.9795 ± 0.0037 | 0.1283 ± 0.0040 | 0.1907 ± 0.0079 | 0.0612 ± 0.0036 | 0.8521 | 0.1275 | – |
| C = A + TS | 0.9786 ± 0.0044 | 0.9785 ± 0.0045 | 0.0125 ± 0.0020 | 0.0610 ± 0.0114 | 0.0349 ± 0.0059 | 0.9759 | 0.0069 | 0.76, 0.94, 0.67, 0.68, 0.75 |
| D = ls0.05 + TS | 0.9777 ± 0.0041 | 0.9777 ± 0.0042 | 0.0122 ± 0.0046 | 0.0621 ± 0.0106 | 0.0340 ± 0.0042 | 0.9765 | 0.0047 | 0.54, 0.56, 0.50, 0.43, 0.51 |
| E 5-seed ensemble | 0.9823 ± 0.0030 | 0.9823 ± 0.0030 | 0.0234 ± 0.0053 | 0.0630 ± 0.0049 | 0.0325 ± 0.0034 | 0.9634 | 0.0189 | – |
| **E + TS (selected)** | **0.9823 ± 0.0030** | **0.9823 ± 0.0030** | **0.0134 ± 0.0023** | **0.0550 ± 0.0095** | **0.0304 ± 0.0052** | 0.9775 | 0.0080 | 0.61, 0.85, 0.68, 0.61, 0.78 |

(ECE/NLL/Brier = per-fold mean ± std, ddof=1; pooled ECE over all 2,200 OOF rows.)

Findings:
- **T < 1 in every fold.** The baseline is under-confident in-distribution, so calibration
  *raises* confidence. The fix is calibration, not lowering numbers.
- **Label smoothing on its own hurts calibration** (ECE 4–7× worse): it adds more
  under-confidence to a model that was already under-confident. With TS on top (D), it ties
  with C.
- The ensemble is the only change that improves accuracy: +0.0037 (≈ 8 rows of 2,200), which is
  within one fold-std of A's spread.
- Selection rule: lowest NLL with no accuracy loss vs A. **E+TS** has the lowest NLL and Brier
  and the highest accuracy. Its per-fold ECE is within noise of C and D. D has the lowest
  pooled ECE but 2 fewer correct rows and higher NLL.

Out-of-range probes (median top-1 probability over the 5 fold models):

| Probe | A | B 0.05 | B 0.1 | C | D | E | E+TS |
|---|---|---|---|---|---|---|---|
| rice_like (in-distribution) | 0.728 | 0.585 | 0.524 | 0.802 | 0.708 | 0.784 | 0.840 |
| all zeros | 0.999 | 0.967 | 0.964 | 1.000 | 1.000 | 0.813 | 0.917 |
| P = 3× max | 1.000 | 0.799 | 0.687 | 1.000 | 0.983 | 0.620 | 0.739 |
| pH = 0 | 0.947 | 0.711 | 0.574 | 0.982 | 0.944 | 0.480 | 0.588 |
| pH = 14 | 1.000 | 0.735 | 0.555 | 1.000 | 0.955 | 0.605 | 0.694 |
| temperature = 60 °C | 1.000 | 0.779 | 0.647 | 1.000 | 0.985 | 0.997 | 0.999 |
| rainfall = 3000 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| humidity = 150% | 1.000 | 0.999 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 |
| all features 2× train max | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

(Full table in `variants.json`.) **No calibration method fixes out-of-range inputs.** TS makes
them slightly *more* confident. The ensemble lowers some of them (pH 0/14, zeros, P) but
rainfall 3000 mm, humidity 150% and 2× max still get 1.000. The input-range guard is therefore
required, not optional.

## Task 1 / step 3 — final model, threshold, guard — 2026-10-25

Command: `python scripts/task1_calibration.py final --variant "E_ensemble5+TS"` (output
`results/task1/final.json`; artifacts in `models/`, git-ignored). Backend TF 2.21 (WSL), 40
epochs. **Training data: the notebook's 80% stratified split (`random_state=42`); the models are
fit on 85% of it (68% of rows) and T on the other 15% (12% of rows). Test = the 20% split, 440
rows, never used for fitting or selection.**

| Model (same split) | Accuracy | Macro-F1 | ECE | NLL | Brier | Mean conf |
|---|---|---|---|---|---|---|
| A (single model, seed 42, no TS), reference | 0.9773 | 0.9771 | 0.0158 | 0.0580 | 0.0321 | 0.9645 |
| **E+TS (deployed), T = 0.6455** | **0.9841** | **0.9840** | **0.0150** | **0.0378** | **0.0221** | 0.9785 |

Single split, so no std. The fold spread is in step 2. ECE on 440 rows is noisy: the two models
differ by 0.0008, well inside the fold std of about 0.002.

**Low-confidence threshold.** Policy: the smallest t on a 0.05 grid whose accepted set
reaches ≥ 99% accuracy on the E+TS out-of-fold predictions. The result is **t = 0.75**. It is
configurable via `M1Recommender(uncertain_threshold=...)`.

| t | OOF coverage | OOF acc (accepted) | OOF acc (flagged) | n flagged |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9823 | – | 0 |
| 0.60 | 0.9873 | 0.9853 | 0.750 | 28 |
| 0.70 | 0.9764 | 0.9874 | 0.769 | 52 |
| **0.75** | **0.9668** | **0.9901** | **0.753** | **73** |
| 0.80 | 0.9600 | 0.9929 | 0.727 | 88 |
| 0.90 | 0.9314 | 0.9951 | 0.808 | 151 |

On the test split at t = 0.75: coverage 0.9659, accepted accuracy 0.9953, 15 of 440 flagged
"uncertain". Flagged predictions are still right about 75% of the time, so "uncertain" means
"check the other two options", not "the model is wrong".

**Input guard.** Per-feature min/max of the 80% training split, widened by 10% of each
feature's range. Values are in `models/crop_config.json` and `final.json`. Any feature outside
these bounds sets status `out_of_range`, which takes priority over `uncertain`.

## Task 1 / step 4 — stage 2 re-trained for the app (notebook protocol, unchanged)

Command: `python scripts/train_stage2_notebook.py`. Backend TF 2.21 (WSL), 40 epochs, 80% of 99
rows (minus Keras `validation_split=0.1`), scaler fit on all 99 rows (the notebook's leaky
protocol, kept on purpose until Tasks 2/4). 3,272 params. **Test accuracy 0.95 on 20 rows (1
error)**, the same as the notebook. Not reliable: there is no baseline or spread yet (Task 4).
Its confidences are not calibrated and the UI says so.

## Task 1 / step 5 — smoke tests via `recommend()` — 2026-10-25

Command: `python scripts/smoke_test.py` (full output: `results/task1/smoke_test.txt`).
Loading the artifacts from disk reproduces the test accuracy in `final.json` exactly (0.9841).

| Input | Status | Top-3 (calibrated) | Fertilizer |
|---|---|---|---|
| rice-like N90 P42 K43 T20.8 H82 pH6.5 R202, Clayey, moisture 45 | ok | rice 89.6%, jute 10.4%, papaya 0.0% | 10-26-26 (70.7%, uncalibrated) |
| pulses-like = chickpea class mean (40.1, 67.8, 79.9, 18.9, 16.9, 7.3, 80.1), Loamy | ok | chickpea 100.0%, kidneybeans 0.0%, grapes 0.0% | none: crop not in fertilizer data (Task 2) |
| all zeros | out_of_range (temperature, humidity, pH) | mothbeans 99.8%, … | – |
| pH = 14, rest rice-like | out_of_range (pH) | jute 99.9%, … | – |
| rainfall = 3000 mm, rest rice-like | out_of_range (rainfall) | rice 100.0%, … | – |
| humidity 150%, temperature 60 °C | out_of_range (temperature, humidity) | papaya 100.0%, … | – |

All-zeros is flagged on temperature, humidity and pH only, because N = 0 is inside the
training range and P = K = 0 fall inside the 10% margin.

---

## Task 2 — fertilizer coverage tiers — 2026-10-25

Command: `python scripts/smoke_test.py` (prints tier coverage). Mapping in `m1/tiers.py`;
the assumption is documented in `docs/fertilizer_tiers.md`. No model or metric changes.

| Tier | Stage-1 crops | Count | Share of 22 crops (= share of stage-1 rows; 100 rows/crop) |
|---|---|---|---|
| 1: ML model, crop in fertilizer data | rice, maize, cotton | 3 | 13.6% |
| 2: ML model via the `pulses` group proxy | chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil | 7 | 31.8% |
| none: "No verified fertilizer data" | apple, banana, coconut, coffee, grapes, jute, mango, muskmelon, orange, papaya, pomegranate, watermelon | 12 | 54.5% |

Total coverage: 10/22 crops (45.5%). Fertilizer-data rows behind each tier: rice 10, maize 6,
cotton 12, pulses 10 (spread across 6 of the 7 fertilizer classes). No end-to-end accuracy is
reported (not computable; see docs). Tier-2 accuracy cannot be measured: the data has no
per-pulse labels.

Smoke test: rice-like → Tier 1 (10-26-26); chickpea class mean → Tier 2 (10-26-26, 99.99%
uncalibrated); mango class mean → no data.

---

## Tasks 3+4 — benchmark (both stages) — 2026-10-25

Command: `python scripts/benchmark.py` (WSL, TF 2.21 backend, LightGBM 4.7.0, scikit-learn 1.9.1;
outputs `results/benchmark/benchmark.json`, `results/benchmark/benchmark_tables.md`). Runtime 231 s.
No hyperparameter search; every model uses defaults or the notebook's settings. Write-up and
conclusions: `docs/architecture.md`.

**Stage 1:** stratified 5-fold (`random_state=42`, same outer folds as Task 1), scaler inside
each fold. **Training data: 100% of each training fold (80% of rows).** MLP: 40 epochs, batch
16, seed 42+fold, single model (no ensemble/TS).

| Model | Accuracy | Macro-F1 | ECE | NLL | Params / tree nodes | Train s/fold | Infer ms/row |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.9709 ± 0.0025 | 0.9707 ± 0.0025 | 0.1288 ± 0.0025 | 0.2102 ± 0.0141 | 176 | 0.12 ± 0.02 | 0.003 |
| RandomForest (300 trees) | 0.9950 ± 0.0030 | 0.9950 ± 0.0030 | 0.0413 ± 0.0029 | 0.0518 ± 0.0018 | 45,484 | 0.96 ± 0.02 | 0.310 |
| LightGBM (defaults) | 0.9914 ± 0.0019 | 0.9913 ± 0.0019 | 0.0083 ± 0.0027 | 0.0312 ± 0.0096 | 70,672 | 0.71 ± 0.10 | 0.042 |
| MLP 64-32 (current) | 0.9805 ± 0.0038 | 0.9804 ± 0.0039 | 0.0166 ± 0.0047 | 0.0581 ± 0.0092 | 3,318 | 11.42 ± 0.21 | 0.502 |

Paired over the 5 folds: RF − MLP accuracy +0.0145 ± 0.0026 (5/5 folds); LightGBM − MLP
+0.0109 ± 0.0054 (5/5); RF − LightGBM +0.0036 ± 0.0041 (4/5). LightGBM NLL < MLP NLL in 5/5
folds. **The tree models beat the MLP on accuracy. LightGBM is also the best calibrated (ECE
and NLL), better even than the deployed MLP ensemble+TS (Task 1: NLL 0.0550 ± 0.0095).**

**Stage 2:** repeated stratified 5-fold × 3 (`random_state=42`), 99 rows, test folds of 19–20
rows. Preprocessing (scaler, one-hot, label encoders) is fit inside each training fold.
**Training data: 80% of rows per split.** NNs: 40 epochs, batch 16, seed 42+split, no
`validation_split`.

| Model | Accuracy | Macro-F1 | ECE | NLL | Params / tree nodes | Train s/fold |
|---|---|---|---|---|---|---|
| Majority class (baseline) | 0.2221 ± 0.0239 | 0.0518 ± 0.0045 | 0.7779 ± 0.0239 | 21.4940 ± 0.6606 | 0 | 0.00 |
| RandomForest (one-hot, 300 trees) | 0.9667 ± 0.0408 | 0.9266 ± 0.0908 | 0.3415 ± 0.0293 | 0.4932 ± 0.0382 | 12,860 | 0.57 ± 0.07 |
| Embedding NN (current) | 0.9330 ± 0.0407 | 0.8451 ± 0.0835 | 0.1839 ± 0.0360 | 0.2803 ± 0.0269 | 3,272 | 4.68 ± 0.77 |
| One-hot NN | 0.8821 ± 0.0700 | 0.8007 ± 0.1317 | 0.1924 ± 0.0476 | 0.4187 ± 0.0699 | 3,783 | 4.23 ± 0.45 |

Paired over the 15 splits (difference ± std; wins/ties/losses for the first model):
- RF − embedding NN: accuracy +0.0337 ± 0.0589 (8/6/1), macro-F1 +0.0815 ± 0.1353 (10/1/4),
  NLL +0.2130 ± 0.0364 (RF worse in 15/15).
- Embedding − one-hot NN: accuracy +0.0509 ± 0.1004 (9/4/2), macro-F1 +0.0444 ± 0.1817
  (7/3/5), NLL −0.1384 ± 0.0600 (embedding better in 15/15).

Mean top-1 confidence: RF 0.643 (accuracy 0.967, so strongly under-confident), embedding NN
0.799, one-hot NN 0.731.

---

## Step 6 — deployed stage 2 retrained without leakage — 2026-10-25

Command: `python scripts/train_stage2.py` (artifacts `models/fert_rf.joblib`,
`models/fert_config.json`; git-ignored). Model: the stage-2 benchmark winner on accuracy and
macro-F1, a RandomForest (300 trees, `random_state=42`) on StandardScaler(numerics) +
OneHot(soil, crop), as one sklearn Pipeline. **Training data: all 99 rows.** Preprocessing is
fit only on the pipeline's own training data. There is no held-out split; the performance
estimate is the benchmark CV (preprocessing fit inside each fold): **accuracy 0.9667 ±
0.0408, macro-F1 0.9266 ± 0.0908, majority-class baseline 0.2221 / 0.0518.** Training-set
accuracy 1.000 is not an estimate. The old notebook NN (`scripts/train_stage2_notebook.py`,
0.95 on 20 rows) is no longer deployed.

**Stage-2 input-range guard** (fertilizer-data min/max ± 10% of range): temperature 25–38,
humidity 50–72, moisture 25–65, N 4–42, K 0–19, P 0–42.

**Range mismatch between datasets** (command: `python scripts/smoke_test.py`):
- Of the 1,000 crop-dataset rows for Tier 1/2 crops, only **1.7%** fall inside the stage-2
  range on all five shared features.
- Per feature inside: temperature 59%, humidity 42%, N 62%, K 46%, P 24%.
- Moisture is not checked, because the crop data has no moisture column.
- So **for almost every realistic stage-1 input, stage 2 extrapolates.** The UI now warns about
  this on each answer.
- Example: on the rice-like smoke test (N 90, K 43, 20.8 °C, 82%), four features are out of
  range. The RF answers Urea at 0.29 confidence; the old NN answered 10-26-26 at 0.71.
- This is a data limitation (the datasets don't overlap), not something a model change fixes.

Smoke tests with the RF: rice-like → Tier 1, Urea 0.29, 4 range warnings; chickpea mean →
Tier 2, Urea 0.27, 4 range warnings; mango mean → no data. Stage-1 outputs unchanged.

---

## SHAP — deployed models — 2026-10-25

Command: `python scripts/shap_summary.py` (plots `results/shap/crop_shap_mean_abs.png`,
`results/shap/fert_shap_mean_abs.png`; numbers `results/shap/shap_summary.json`).

**Crop model:** KernelExplainer on the deployed ensemble+TS (calibrated probabilities, raw
input units). Background = k-means(20) of the 80% training split; 220 test rows explained
(10 per class); nsamples 256. Mean |SHAP| averaged over rows and all 22 classes:

| humidity | rainfall | N | P | K | temperature | pH |
|---|---|---|---|---|---|---|
| 0.0395 | 0.0356 | 0.0317 | 0.0287 | 0.0236 | 0.0123 | 0.0059 |

**Fertilizer model:** exact TreeExplainer on the deployed RandomForest. One-hot columns are
summed back to `soil_type` and `crop`. 99 rows explained (all of its training data). Mean
|SHAP| averaged over rows and all 7 classes:

| P | N | K | crop | Moisture | temperature | soil_type | humidity |
|---|---|---|---|---|---|---|---|
| 0.0772 | 0.0678 | 0.0629 | 0.0177 | 0.0116 | 0.0110 | 0.0101 | 0.0092 |

Reading:
- **Fertilizer:** the choice is driven by the soil's N/P/K levels. Crop and soil type
  contribute little. This fits fertilizer names that are NPK grades, and it is one more reason
  the Tier-2 proxy only matters at the margin.
- **Crop:** humidity, rainfall and N/P/K drive the crop model; pH matters least. The
  notebook's one-class SHAP (K, rainfall, humidity on top) was consistent but partial.

## Gradio app — final state — 2026-10-25

`python app.py [--share] [--host 0.0.0.0 --port 7860]`. It loads `models/` (no retraining) and
calls `m1.recommender.recommend()`. Outputs: top-3 crops with calibrated probabilities; status
(ok / uncertain < 0.75 / out_of_range with the offending features); fertilizer with its tier
label, uncalibrated confidence and stage-2 range warnings; per-prediction SHAP reasons (Kernel
SHAP on the top crop, nsamples 128; TreeSHAP on the fertilizer). Latency in WSL: 0.10–0.49 s per
request. Headless check: page served with HTTP 200. Full smoke output:
`results/task1/smoke_test.txt`.

The fertilizer reasons for the rice-like and chickpea-like inputs are identical
(N +0.24, P −0.10, K −0.07). Both inputs exceed the RF's highest N/P/K split points, so they fall
into the same leaves. This is another symptom of the stage-2 range mismatch.

---

## Decision: neural networks for both stages — 2026-10-25

Project-owner decision (scope is deep learning). Stage 1 stays the MLP ensemble + TS (Task 1).
Stage 2 is switched back to the embedding NN. **RandomForest and LightGBM are kept as
benchmark results only.** Full reasoning: `docs/architecture.md`.

What the benchmark shows, stated plainly:
- **Stage 1: tree models beat the NN in all 5 folds.** RF +0.0145 and LightGBM +0.0109
  accuracy over the single MLP, several times the fold std. LightGBM is also better calibrated
  (NLL 0.031 vs 0.055 for the deployed ensemble+TS). Keeping the NN costs about 1.3 points of
  accuracy (deployed 0.9823 vs RF 0.9950 in CV).
- **Stage 2: RF vs embedding NN is within noise.** The paired differences (accuracy
  +0.034 ± 0.059, macro-F1 +0.082 ± 0.135) are smaller than their split-to-split std, with
  ~20-row test folds and overlapping splits. RF was ahead or tied in 14 of 15 splits on
  accuracy, so the direction favours RF but the size is not measurable with 99 rows. The NN has
  the better NLL in 15 of 15 splits.

## Step 6 (revised) — deployed stage 2 = embedding NN, leak-free — 2026-10-25

**Supersedes the RandomForest deployment above.** Command: `python scripts/train_stage2.py`
(artifacts `models/fert_nn.keras`, `models/fert_nn_preprocessing.pkl`, `models/fert_config.json`;
git-ignored).
- **Model:** notebook embedding NN (3,272 params), code moved unchanged to `m1/stage2.py`.
  The scaler and encoders are fit inside `FertNN.fit`.
- **Training data: all 99 rows.** 40 epochs, batch 16, seed 42.
- **Check:** the moved code reproduces benchmark split 1 exactly (accuracy 0.85, NLL 0.3186).
- **Estimate:** the benchmark CV, **accuracy 0.9330 ± 0.0407, macro-F1 0.8451 ± 0.0835,
  ECE 0.1839 ± 0.0360**, against a majority-class baseline of 0.2221 / 0.0518. Training
  accuracy 1.000 is not an estimate.
- **Guard:** the same stage-2 input-range guard as before.

Smoke tests (`python scripts/smoke_test.py`, output `results/task1/smoke_test.txt`):

| Input | Fertilizer (NN) | Confidence | Tier | Stage-2 range warnings |
|---|---|---|---|---|
| rice-like | 10-26-26 | 0.85 | 1 | temperature, humidity, N, K |
| chickpea class mean | 10-26-26 | 1.00 | 2 | temperature, humidity, K, P |
| mango class mean | – (no data) | – | none | – |

**Extrapolation behaviour:** unlike the RF (≈0.27 confidence on the same inputs), the NN is
highly confident far outside its training range. The chickpea answer is at 100%, with SHAP
attributing +0.95 to K = 79.9 (training max 19). The range warning is the only safeguard here.

## SHAP — fertilizer model re-run for the deployed NN — 2026-10-25

Command: `python scripts/shap_summary.py` (plot `results/shap/fert_shap_mean_abs.png`; numbers
`results/shap/shap_summary.json`). KernelExplainer on the deployed embedding NN. Inputs are the
6 numerics + soil type + crop, with each categorical as a single feature. Background = 20 real
rows (seed 42); nsamples 256 (exact enumeration for 8 features); 99 rows explained. Mean
|SHAP| over rows and all 7 classes:

| P | N | K | temperature | crop | Moisture | humidity | soil_type |
|---|---|---|---|---|---|---|---|
| 0.1164 | 0.1110 | 0.0895 | 0.0119 | 0.0090 | 0.0061 | 0.0043 | 0.0025 |

Same picture as the RF: N/P/K drive the fertilizer choice; crop and soil type contribute
little. The crop-model SHAP numbers are unchanged (re-run reproduced them exactly). The app's
per-prediction fertilizer reason now uses Kernel SHAP on the NN (nsamples 256). App latency is
0.10–0.64 s, and the headless check returned HTTP 200.
