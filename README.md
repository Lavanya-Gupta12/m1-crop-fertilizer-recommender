# M1 — Crop & Fertilizer Recommender

Module M1 of the Smart Crop Advisory System (SIH25010).

**Pipeline**
- **Stage 1** (neural network ensemble, temperature-scaled): soil and climate → the top-3 crops
  with calibrated probabilities.
- **Stage 2** (embedding neural network): soil, climate and the predicted crop → a fertilizer,
  labelled with the tier it came from.
- **Every answer includes** input-range warnings, an "uncertain" flag and SHAP reasons.

**Run it in Colab:** [notebooks/M1_colab.ipynb](https://colab.research.google.com/github/Lavanya-Gupta12/m1-crop-fertilizer-recommender/blob/main/notebooks/M1_colab.ipynb). It's a self-contained walkthrough with all model code inline. `QUICK_MODE` gives a fast demo; full mode reproduces `RESULTS.md`.

**Report:** [`notebooks/M1_final.ipynb`](notebooks/M1_final.ipynb), also as
[HTML](notebooks/M1_final.html) and [PDF](notebooks/M1_final.pdf).

**Further documents**
- [`RESULTS.md`](RESULTS.md): every number, with the command that produced it.
- [`docs/architecture.md`](docs/architecture.md): design choices and the benchmark against tree models.
- [`docs/fertilizer_tiers.md`](docs/fertilizer_tiers.md): fertilizer coverage and the Tier-2 assumption.

## Headline results (mean ± std over folds)

| | Accuracy | Macro-F1 | Baseline |
|---|---|---|---|
| Stage 1, deployed ensemble + TS (5-fold CV) | 0.9823 ± 0.0030 | 0.9823 ± 0.0030 | class prior 0.0455 |
| Stage 2, embedding NN (repeated 5-fold × 3) | 0.9330 ± 0.0407 | 0.8451 ± 0.0835 | majority class 0.2221 |

Stage 1 calibration: ECE 0.0193 → 0.0134 and NLL 0.0663 → 0.0550 after the ensemble and
temperature scaling.

## Known limitations
- **Tree models beat the stage-1 NN in all 5 folds** (RandomForest 0.9950). The NN is kept
  because the project scope is deep learning.
- **Fertilizer coverage is 10 of 22 crops.** Tier 2 (the pulses proxy) is an assumption.
- **The two datasets barely overlap.** Only 1.7% of realistic inputs fall inside the
  fertilizer data's range, so stage 2 usually extrapolates, and the app warns when it does.
- **End-to-end accuracy can't be measured** with this data.

## Run

Tested on WSL2 Ubuntu 24.04, Python 3.12.3, TensorFlow 2.21 (CPU). LightGBM, which the
benchmark uses, also needs `sudo apt install -y libgomp1`.

```bash
python -m venv ~/venvs/m1 && . ~/venvs/m1/bin/activate && pip install -r requirements.txt
python scripts/task1_calibration.py baseline && python scripts/task1_calibration.py variants
python scripts/task1_calibration.py final --variant "E_ensemble5+TS"
python scripts/train_stage2.py && python scripts/shap_summary.py && python scripts/smoke_test.py
python app.py            # add --share for a temporary public link, or --host 0.0.0.0 to embed
```

The data is downloaded with `kagglehub` (`atharvaingle/crop-recommendation-dataset`,
`gdabhishek/fertilizer-prediction`). Alternatively, put the two CSVs in `data/`. Trained models
are written to `models/`, which is not committed.
