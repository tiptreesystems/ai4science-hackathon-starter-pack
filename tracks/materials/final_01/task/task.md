# CO₂RR Electrocatalysts — Descriptor Engineering for Faradaic Efficiency

Engineer chemical descriptors for metal-alloy electrocatalysts under CO₂ reduction conditions. The scorer fits a fixed RandomForestRegressor on your descriptors and rewards how much variance in Faradaic efficiency to CO your features explain on a held-out set of alloys.

The model is fixed: the scorer trains a `RandomForestRegressor` (scikit-learn defaults, `n_estimators=100`, averaged over 5 seeds). Your task is to derive a feature set — using cheminformatics / materials-informatics descriptors, force-field or ML-force-field calculations, semi-empirical electronic-structure methods, atomic-property aggregates, fingerprints, embeddings, ratios, etc. — that, when fed into that fixed model, explains as much of the held-out target as possible. See the **Compute budget** section below for which methods fit the hackathon timeline.

## Files

- `data/train.csv`: 162 labeled compositions with `id`, 13 element columns (Ag, Au, Cd, Cu, Ga, Hg, In, Ni, Pd, Rh, Sn, Tl, Zn), and the observed target `fe_co`.
- `data/test.csv`: 52 held-out compositions with `id` and the same element columns (labels withheld).
- `sample_train_features.csv`, `sample_test_features.csv`: example submission shape using the raw element columns. Replace these with your engineered features.

## Inputs

- `id`: unique row identifier (string).
- `Ag` … `Zn` (13 element columns: Ag, Au, Cd, Cu, Ga, Hg, In, Ni, Pd, Rh, Sn, Tl, Zn). Mole fractions per row sum to 1.0.
- `fe_co` (train only): observed Faradaic efficiency to CO (fraction). Train targets span [0, 0.455]; held-out targets skew slightly higher (mean 0.130 vs train mean 0.095).

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

For reference, the same random forest fit on the raw element mole-fraction / loading columns (no descriptor engineering) achieves R² ≈ 0.3031 on this split. The leaderboard `score` is the mean R² clipped to `[0, 1]`.

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

R² is computed in the native [0, 1] target space. R² is unbounded below; a feature set worse than predicting the train mean produces a negative R² and a floor-zero leaderboard score.
