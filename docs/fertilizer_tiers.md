# Fertilizer coverage tiers (Task 2)

Stage 1 predicts 22 crops; the fertilizer dataset (99 rows) labels 11 crops. They share 3
names. Every fertilizer answer states which tier produced it. Mapping: `m1/tiers.py`.

| Tier | Source | Stage-1 crops | Count |
|---|---|---|---|
| 1 | ML model, crop is in the fertilizer data | rice, maize, cotton | 3 |
| 2 | ML model queried with the crop-group label `pulses` | chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil | 7 |
| – | none: "No verified fertilizer data for this crop." | apple, banana, coconut, coffee, grapes, jute, mango, muskmelon, orange, papaya, pomegranate, watermelon | 12 |

Coverage: 10 of 22 crops (45%). The crop dataset is balanced (100 rows per crop), so this is also
45% of stage-1 training rows. The dataset's other labels (barley, groundnuts, millets, oilseeds,
sugarcane, tobacco, wheat) have no stage-1 counterpart, so stage 1 never predicts them.

## Tier 2 assumption

- The fertilizer dataset uses one class, `Pulses`, for all pulse crops. The seven stage-1 crops
  above are all pulses (grain legumes), so they are mapped to that class.
- **Assumption:** the fertilizer choice the dataset records for "pulses" applies equally to each
  individual pulse crop under the same soil, moisture and N/P/K readings. Agronomically this is
  the coarsest defensible grouping (shared legume habit and N-fixation). It ignores
  crop-specific differences in nutrient demand.
- **Evidence is thin:** the `pulses` class has 10 rows covering 6 different fertilizers, and
  none of those rows says which pulse they came from. The proxy cannot be validated against
  crop-specific labels, and no accuracy is claimed for it.
- Groupings that were **not** made, because they are not defensible: coconut → oilseeds
  (plantation palm, not an oilseed crop in the dataset's sense), jute → anything, and any
  fruit crop → any field-crop label.

## What is not claimed
- No end-to-end accuracy: there are no rows with both stage-1 inputs (pH, rainfall) and a
  fertilizer label, so stage-1 → stage-2 accuracy cannot be measured. Each stage is reported
  on its own, plus the coverage above.
- No dosage: the dataset gives fertilizer *names*, not amounts. The PAU/ICAR dose table is out
  of scope for now.
