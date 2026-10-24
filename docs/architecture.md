# M1 architecture and hyperparameters (Tasks 3 + 4)

Each choice is marked as one of:
- **[default]**: a reasonable starting value, never tested against alternatives.
- **[validated]**: compared against alternatives under the protocol below; evidence linked.
- **[reasoned]**: follows from the data or the maths, not from an experiment.

No hyperparameter search was run (dropped from scope for time), so everything marked
[default] is still untuned.

Benchmark protocol (`scripts/benchmark.py`, results in `results/benchmark/`, logged in
`RESULTS.md`): stage 1 uses stratified 5-fold (`random_state=42`); stage 2 uses repeated
stratified 5-fold × 3 (n = 99). Preprocessing (scaler, one-hot) is fit inside each training
fold, seeds are fixed, and std uses ddof = 1.

## Stage 1: crop model (7 → Dense 64 ReLU → Dropout 0.2 → Dense 32 ReLU → Softmax 22)

| Choice | Status | Why |
|---|---|---|
| Standardised inputs (`StandardScaler`) | [reasoned] | Features span very different scales (pH 3.5–9.9 vs rainfall 20–299 mm). Gradient-based NNs with a single learning rate are scale-sensitive. Now fit inside each training fold; the old pre-split fit changed CV accuracy by only −0.0005 (Task 1). |
| Stratified split / folds | [reasoned] | 22 balanced classes (100 rows each). Stratification keeps every fold balanced, so per-fold macro-F1 is comparable. |
| 64 → 32 funnel | [default] | The last hidden layer (32) is ≥ the number of classes (22), so it is not a bottleneck before the softmax. Widths were never tuned. |
| Model size: 3,318 params | [reasoned] | A training fold has 1,760 rows (1,584 in the notebook, after Keras' `validation_split=0.1`). A few thousand parameters is small for that and gives a low overfitting risk. Train and validation accuracy stayed close in the notebook (≈0.975 / 0.977 at epoch 40). |
| Dropout 0.2 | [default] | Light regularisation. Not compared against 0 or higher values. |
| Adam, lr 1e-3 | [default] | Keras default. Not tuned. |
| Batch 16 | [default] | 1,584 / 16 = 99 steps/epoch in the notebook (110 on a full 1,760-row fold; 94 on the 85% Task-1 fit split). Not tuned. |
| 40 epochs, no EarlyStopping | [default, weak evidence] | The notebook used 25 epochs for CV and 40 for the final model. Task 1 used 40 everywhere for consistency. 40 epochs on 85% of a fold beat 25 epochs on 100% (0.9786 vs 0.9727), suggesting 25 under-trains. There is no EarlyStopping or epoch search. |
| Softmax + cross-entropy | [reasoned] | Standard for single-label multiclass. Gives a probability per crop, so a ranked top-3 is possible. |
| 5-seed ensemble + temperature scaling (deployed) | **[validated]** | Task 1: lowest NLL (0.0550 vs 0.0663) with no accuracy loss. T is fit on a held-out calibration split. Label smoothing was tested and rejected. |
| No soil-type input | [reasoned] | The crop dataset has no soil column (deviation from the deck). |

### Benchmark: does the MLP earn its place? (stage 1, 5-fold, mean ± std)

Trees are benchmark-only; the deployed stage-1 model is the MLP ensemble + TS (last row).

| Model | Accuracy | Macro-F1 | ECE | NLL | Params / tree nodes | Train s/fold | Inference ms/row | SHAP |
|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.9709 ± 0.0025 | 0.9707 ± 0.0025 | 0.1288 ± 0.0025 | 0.2102 ± 0.0141 | 176 | 0.12 | 0.003 | Linear (exact) |
| RandomForest (300 trees) | **0.9950 ± 0.0030** | **0.9950 ± 0.0030** | 0.0413 ± 0.0029 | 0.0518 ± 0.0018 | 45,484 nodes | 0.96 | 0.310 | Tree (exact) |
| LightGBM (defaults) | 0.9914 ± 0.0019 | 0.9913 ± 0.0019 | **0.0083 ± 0.0027** | **0.0312 ± 0.0096** | 70,672 nodes | 0.71 | 0.042 | Tree (exact) |
| MLP 64-32 (current, single model) | 0.9805 ± 0.0038 | 0.9804 ± 0.0039 | 0.0166 ± 0.0047 | 0.0581 ± 0.0092 | 3,318 | 11.42 | 0.502* | Kernel (approximate) |
| *MLP ensemble + TS (deployed; Task 1, trained on 85% of each fold)* | 0.9823 ± 0.0030 | 0.9823 ± 0.0030 | 0.0134 ± 0.0023 | 0.0550 ± 0.0095 | 5 × 3,318 | ~57 | – | Kernel |

\*Keras `predict()` overhead on one small batch, not compute cost.

**Conclusion (stage 1): the NN does not win on accuracy or calibration.**
- **Accuracy:** RandomForest and LightGBM beat the MLP in **5 of 5 folds**, by +1.45 and +1.09
  points. The gap is several times the fold std, so it is not noise.
- **Calibration:** LightGBM, untuned, is better calibrated than even the deployed
  ensemble+TS (NLL 0.031 vs 0.055; lower than the single MLP in 5 of 5 folds).
- **What the NN does offer:** a much smaller model (3.3k parameters vs 45–70k tree nodes), one
  architecture for both stages, and well-behaved probabilities after temperature scaling.

**Decision (project owner, 2026-10-25): keep the NN for both stages.**
- The project scope is deep learning, so both stages stay neural networks. Stage 1 stays the
  5-seed MLP ensemble + temperature scaling from Task 1.
- RandomForest and LightGBM remain **benchmark results only**: they show what the data
  supports and are reported, not hidden.
- Cost of this decision: about 1–1.5 points of stage-1 accuracy (deployed ensemble 0.9823 vs
  RF 0.9950 in CV) and approximate Kernel SHAP instead of exact TreeSHAP.
- What it does not cost: calibration is handled (Task 1), and the range guard and uncertainty
  flag work the same for any model.

## Stage 2: fertilizer model (numeric 6 + soil Embedding(5→2) + crop Embedding(11→5) → 64 → 32 → Softmax 7)

| Choice | Status | Why |
|---|---|---|
| Embedding dim = n // 2 | [default] | Heuristic, not tuned. |
| Embeddings vs one-hot inputs | **[validated, weakly]** | See the ablation below. Embeddings beat one-hot on NLL in 15/15 splits and on accuracy 9/4/2 (wins/ties/losses); macro-F1 is mixed (7/3/5). |
| Same 64-32 head, dropout, Adam, batch 16, 40 epochs as stage 1 | [default] | Chosen for consistency. Large for 79 training rows (3,272 params). |

### Benchmark (stage 2, repeated stratified 5-fold × 3, test folds ≈ 20 rows, mean ± std)

| Model | Accuracy | Macro-F1 | ECE | NLL | Params / tree nodes | Train s/fold |
|---|---|---|---|---|---|---|
| Majority class (baseline) | 0.2221 ± 0.0239 | 0.0518 ± 0.0045 | 0.7779 ± 0.0239 | 21.49 ± 0.66 | 0 | 0.00 |
| RandomForest (one-hot, 300 trees) | **0.9667 ± 0.0408** | **0.9266 ± 0.0908** | 0.3415 ± 0.0293 | 0.4932 ± 0.0382 | 12,860 nodes | 0.57 |
| **Embedding NN (current, deployed)** | 0.9330 ± 0.0407 | 0.8451 ± 0.0835 | **0.1839 ± 0.0360** | **0.2803 ± 0.0269** | 3,272 | 4.68 |
| One-hot NN (ablation) | 0.8821 ± 0.0700 | 0.8007 ± 0.1317 | 0.1924 ± 0.0476 | 0.4187 ± 0.0699 | 3,783 | 4.23 |

**Conclusion (stage 2): the differences between RandomForest and the embedding NN are within
noise.**
- **Size of the gap:** RF − embedding NN is accuracy +0.034 ± 0.059 and macro-F1
  +0.082 ± 0.135 (paired over 15 splits). Each mean difference is smaller than its
  split-to-split std.
- **Why it can't be settled:** the test folds have ~20 rows (one row = 5 accuracy points), and
  the 15 splits reuse the same 99 rows, so they are not independent.
- **Direction, stated for the record:** RF was ahead or tied on accuracy in 14 of 15 splits.
  The direction favours RF, but the size of the gap cannot be told apart from noise with this
  data.
- **Probabilities:** the embedding NN has the better NLL in 15 of 15 splits (0.28 vs 0.49), and
  RF is strongly under-confident (mean confidence 0.64 vs accuracy 0.97).
- **Embeddings vs one-hot:** the earlier guess that one-hot "may do equally well" was wrong for
  this NN. Embeddings are better on NLL in every split and usually on accuracy (9/4/2).

**Deployed: the embedding NN** (project owner decision, deep-learning scope). It is retrained
leak-free: the scaler and encoders are fit inside `FertNN.fit`, on all 99 rows. Its estimate is
the CV above (accuracy 0.933 ± 0.041, macro-F1 0.845 ± 0.084), against a majority-class baseline
of 0.222. RF is kept as a benchmark result only.

**Caveat that matters more than the model choice.** Only 1.7% of realistic stage-1 inputs for
Tier 1/2 crops fall inside the fertilizer data's range. Outside that range the NN extrapolates
with *high* confidence: the chickpea smoke test gives 10-26-26 at 100%, driven by K = 79.9
against a training max of 19. The RF plateaued at ~0.27 on the same input. The stage-2 range
guard and the extrapolation warning in the UI are therefore essential, not cosmetic.
