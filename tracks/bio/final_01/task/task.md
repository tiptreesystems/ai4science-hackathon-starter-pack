# Protein Thermostability Mutation Prediction

Predict dTm, the change in protein melting temperature in degrees C,
caused by a single-point amino-acid substitution. Positive values are
stabilizing and negative values are destabilizing.

The split is protein-cold: every protein in the test set is absent from
training. A useful model should generalize across protein families rather
than memorize protein-specific mutation effects.

## Files

- `data/train.csv`: labeled mutations with `id`, `protein_id`, `sequence`,
  `position`, `wt_aa`, `mut_aa`, and `dTm`.
- `data/test.csv`: unlabeled mutations with the same columns except `dTm`.
- `sample_submission.csv`: valid output format for every test ID.

## Output

Write `predictions.csv` with exactly these columns:

```csv
id,prediction
test_00000,-1.25
```

`prediction` must be a finite real-valued dTm estimate in degrees C.
Include every test `id` exactly once and no extra rows.

## Metric

The scorer computes Spearman rank correlation between predicted and true
dTm on hidden test mutations. Higher is better. Constant predictions score
zero.

## Notes

`position` is one-indexed and satisfies `sequence[position - 1] == wt_aa`.

