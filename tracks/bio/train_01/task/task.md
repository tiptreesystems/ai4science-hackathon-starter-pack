# Enzyme-Substrate Activity Prediction

Predict whether a halogenase enzyme will convert a candidate small-molecule
substrate in the presence of NaBr. Each row is an enzyme protein sequence
and substrate SMILES pair.

The split is substrate-cold: every substrate in the test set is unseen in
training. Every test enzyme appears in the training data. A useful model
should therefore generalize across substrate chemistry rather than memorize
substrate identity.

## Files

- `data/train.csv`: labeled training pairs with `id`, `sequence`,
  `substrate`, and binary `activity`.
- `data/test.csv`: unlabeled test pairs with `id`, `sequence`, and
  `substrate`.
- `sample_submission.csv`: valid output format for every test ID.

## Output

Write `predictions.csv` with exactly these columns:

```csv
id,prediction
test_00000,0.25
```

`prediction` should be a finite probability in `[0, 1]` that the pair is
active. Include every test `id` exactly once and no extra rows.

## Metric

The scorer computes area under the precision-recall curve (AUPRC) against
hidden labels. Higher is better.

## Notes

The training labels are imbalanced, with roughly 12% positives. The test
split preserves this rate while holding out substrates.

