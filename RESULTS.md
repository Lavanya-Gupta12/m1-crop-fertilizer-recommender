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
