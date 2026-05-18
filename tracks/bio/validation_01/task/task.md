# Peptide-MHC Class I Binding Affinity Prediction

## Problem

Given a 9-residue candidate peptide and an HLA class I allele, predict how
tightly the peptide binds the allele. The split is **allele-cold**: six HLA
alleles are absent from the training set, and the entire test set is drawn from
those held-out alleles. A successful model must generalize across the
peptide-binding groove rather than memorizing per-allele preferences.

This is the pan-allele regime that MHCflurry and NetMHCpan were designed to
tackle — predicting affinity for an allele the model has never seen during
training, by leaning on the 34-residue HLA pseudo-sequence.

## Input format

`data/train.csv` columns:

| column              | type  | description |
|---------------------|-------|-------------|
| `id`                | str   | Unique identifier for the (peptide, allele) pair, e.g. `train_00037`. |
| `peptide`           | str   | Nine-residue candidate peptide (single-letter amino acids, 20 standard residues only). |
| `allele`            | str   | HLA class I allele in NetMHCpan-style four-digit form (e.g. `HLA-A0201`). |
| `allele_pseudo_seq` | str   | 34-residue HLA pseudo-sequence (NetMHCpan convention) covering the peptide-binding groove. Shipped so participants do not need an HLA database lookup. |
| `target`            | float | Transformed binding affinity in `[0, 1]`: `1 - log(IC50_nM) / log(50000)`. Higher means tighter binding. |

`data/test.csv` has the same columns minus `target`. Every test `allele` is one
of the six held-out alleles (not present in `train.csv`).

## Output format

Write `predictions.csv` with exactly these columns:

| column       | type  | description |
|--------------|-------|-------------|
| `id`         | str   | Must match an `id` from `test.csv`, exactly once each. |
| `prediction` | float | Predicted transformed affinity in `[0, 1]`. |

Order is not required; all test `id`s must be present and no extras.

## Scoring

The metric is **Spearman rank correlation** between predicted and reference
`target` on the held-out pairs, computed in pure Python with average-rank tie
handling. Higher is better; predictions that are constant or all-equal score 0.
The submission is normalized against a baseline Spearman of `metadata.baseline`
(simple BLOSUM62 + MLP trained on the same allele-cold split) and an ideal of
`1.0`.

## Split note

The split is allele-cold by design: six HLA alleles
(`HLA-A0203, HLA-A0301, HLA-A2402, HLA-B0702, HLA-B1501, HLA-B5701`) are
held out entirely. Each held-out allele has at least one closely related
same-locus sibling allele in the training set (e.g. `HLA-A0301` is held out
but `HLA-A1101`, `HLA-A6801`, `HLA-A3101` from the A3 supertype remain
in training), so the test is OOD without being adversarial.

## Source

Affinity measurements come from MHCflurry's `data_curated` release
(`curated_training_data.affinity.csv`, October 2023), which is derived from
IEDB and standardized. The 34-residue HLA pseudo-sequence per allele comes from
the MHCflurry `class1_pseudosequences.csv` and follows the NetMHCpan
convention.
