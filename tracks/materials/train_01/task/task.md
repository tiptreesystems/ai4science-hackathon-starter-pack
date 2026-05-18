# DRM Catalyst — Descriptor Engineering for Methane Conversion

Engineer chemical and materials-informatics descriptors for multi-metal catalysts operating under dry reforming of methane (DRM). The scorer fits a fixed RandomForestRegressor on your descriptors and rewards how much methane-conversion variance your features explain on a held-out set of compositions.

The model is fixed: the scorer trains a `RandomForestRegressor` (scikit-learn defaults, `n_estimators=100`, averaged over 5 seeds). Your task is to derive a feature set — using cheminformatics / materials-informatics descriptors, force-field or ML-force-field calculations, semi-empirical electronic-structure methods, atomic-property aggregates, fingerprints, embeddings, ratios, etc. — that, when fed into that fixed model, explains as much of the held-out target as possible. See the **Compute budget** section below for which methods fit the hackathon timeline.

## Files

- `data/train.csv`: 224 labeled compositions with `id`, 17 element columns (Li, Al, Ca, V, Ni, Nb, Rh, Ag, Sn, Cs, Ba, Ce, Sm, Hf, Ir, Au, Bi), and the observed target `CH4_conv_pct`.
- `data/test.csv`: 64 held-out compositions with `id` and the same element columns (labels withheld).
- `sample_train_features.csv`, `sample_test_features.csv`: example submission shape using the raw element columns. Replace these with your engineered features.

## Inputs

- `id`: unique row identifier (string).
- `Li` … `Bi` (17 element columns: Li, Al, Ca, V, Ni, Nb, Rh, Ag, Sn, Cs, Ba, Ce, Sm, Hf, Ir, Au, Bi). Per-element loadings are raw values (row sums vary around 2.7-2.8) and are not normalised to 1.
- `CH4_conv_pct` (train only): observed methane conversion (percent of CH₄ consumed). Train targets span roughly [-0.2, 40]; small negative values come from measurement noise around zero conversion.

## Submission

Write **two** CSV files. Both must use `id` as the first column and otherwise contain only finite numeric feature columns. The two files must share **exactly the same feature column names in the same order**; any number of feature columns is allowed.

```csv
# train_features.csv
id,my_feat_1,my_feat_2,...
train_00000,1.23,0.45,...
...
```

```csv
# test_features.csv
id,my_feat_1,my_feat_2,...
test_00000,2.34,0.56,...
...
```

Every train `id` from `data/train.csv` must appear exactly once in `train_features.csv`; every test `id` from `data/test.csv` must appear exactly once in `test_features.csv`; no extras.

## Metric

The scorer fits `RandomForestRegressor(n_estimators=100, random_state=seed)` (sklearn defaults otherwise) on your `train_features.csv` joined to the hidden train labels, then predicts on `test_features.csv` and computes R² (`sklearn.metrics.r2_score`) against the hidden held-out labels. The process is repeated for seeds 0–4 and the mean R² is reported as `metric_value`. Higher is better; R² = 0 means "no skill" (matching predict-the-mean), R² = 1 is perfect, and R² is unbounded below.

For reference, the same random forest fit on the raw element mole-fraction / loading columns (no descriptor engineering) achieves R² ≈ 0.4130 on this split. The leaderboard `score` is the mean R² clipped to `[0, 1]`.

## Compute budget

Plain DFT (VASP, Quantum ESPRESSO, ORCA at hybrid-functional level, etc.) is too slow for the hackathon timeline — a single self-consistent calculation on a realistic unit cell can take hours per composition. Stay within methods that return per-composition descriptors in seconds to minutes:

- **Tabular / cheminformatics descriptors** (Magpie, Matminer, pymatgen, RDKit) — fastest option and a strong starting point.
- **Classical force fields** (LAMMPS, GROMACS, ASE-bundled potentials) for structural and energetic estimates when a parameterisation is available.
- **Pre-trained universal ML force fields** (e.g. MACE-MP-0, CHGNet, M3GNet, ALIGNN-FF) for energies, forces, and relaxed-structure descriptors across the periodic table at ~seconds per single-point evaluation.
- **Semi-empirical electronic structure** such as **g-xTB** or GFN-xTB (via the `xtb` / `tblite` packages) for single-point electronic descriptors (HOMO/LUMO, dipoles, partial charges, polarisability) at ~seconds per composition.

Heavier first-principles calculations are not forbidden — just budget your time. If you already have DFT results for these compositions in a public dataset, you may use them.

## Rules

- The model is fixed by the scorer; do not submit predictions, model weights, or pretrained model outputs as features unless they are deterministic functions of the composition.
- Use any public software for descriptor generation and any public reference data (atomic-property tables, Materials Project records, OQMD, etc.). Do not consult hidden reference files or external labels for these specific compositions.
- Feature values must be finite (no NaN, ±inf).

## Notes

The split is provided as-is from the source repository. The held-out test set is skewed toward lower-activity catalysts (test mean ~5.6 vs train mean ~8.8), so good descriptors must generalise to less active compositions, not just interpolate the training distribution.
