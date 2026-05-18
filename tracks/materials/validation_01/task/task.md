# Solid Electrolytes — Descriptor Engineering (Validation Phase)

Validation phase of the solid-electrolyte descriptor engineering task. Engineer chemical / structural descriptors; the scorer fits a fixed RandomForestRegressor on log10-conductivity and rewards orders-of-magnitude variance explained on held-out compositions.

The model is fixed: the scorer trains a `RandomForestRegressor` (scikit-learn defaults, `n_estimators=100`, averaged over 5 seeds). Your task is to derive a feature set — using cheminformatics / materials-informatics descriptors, force-field or ML-force-field calculations, semi-empirical electronic-structure methods, atomic-property aggregates, fingerprints, embeddings, ratios, etc. — that, when fed into that fixed model, explains as much of the held-out target as possible. See the **Compute budget** section below for which methods fit the hackathon timeline.

## Files

- `data/train.csv`: 400 labeled compositions with `id`, 55 element columns (Ag, Al, B, Ba, Bi, Br, C, Ca, Cd, Ce, Cl, Co, Cr, Cu, Er, F, Fe, Ga, Gd, Ge, H, Hf, I, In, K, La, Li, Lu, Mg, Mn, Mo, N, Na, Nb, Nd, O, P, Pb, Pr, S, Sb, Sc, Se, Si, Sm, Sn, Sr, Ta, Te, Ti, V, W, Y, Zn, Zr), and the observed target `ionic_conductivity_S_cm`.
- `data/test.csv`: 100 held-out compositions with `id` and the same element columns (labels withheld).
- `sample_train_features.csv`, `sample_test_features.csv`: example submission shape using the raw element columns. Replace these with your engineered features.

## Inputs

- `id`: unique row identifier (string).
- `Ag` … `Zr` (55 element columns: Ag, Al, B, Ba, Bi, Br, C, Ca, Cd, Ce, Cl, Co, Cr, Cu, Er, F, Fe, Ga, Gd, Ge, H, Hf, I, In, K, La, Li, Lu, Mg, Mn, Mo, N, Na, Nb, Nd, O, P, Pb, Pr, S, Sb, Sc, Se, Si, Sm, Sn, Sr, Ta, Te, Ti, V, W, Y, Zn, Zr). Mole fractions per row sum to 1.0.
- `ionic_conductivity_S_cm` (train only): observed room-temperature ionic conductivity (S/cm). Conductivity spans roughly 17 orders of magnitude (~1e-18 to ~1e-2 S/cm). The scorer trains the random forest on log10(target) and reports R² in log10 space.

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

Conductivity spans ~17 orders of magnitude, so the random forest is trained on `log10(target)` (with targets clipped at 1e-20). The scorer fits `RandomForestRegressor(n_estimators=100, random_state=seed)` (sklearn defaults otherwise) on your `train_features.csv` joined to log10 train labels, predicts log10-conductivity on `test_features.csv`, and computes R² in log10 space against the hidden held-out log labels. The process is repeated for seeds 0–4 and the mean log10-R² is reported as `metric_value`. Higher is better; log10-R² = 0 means "no skill" (matching predict-the-log-mean) and log10-R² = 1 is perfect.

For reference, the same random forest fit on the raw element mole-fraction columns (no descriptor engineering) achieves log10-R² ≈ 0.7732 on this split. The leaderboard `score` is the mean log10-R² clipped to `[0, 1]`.

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

Targets at or below 1e-20 S/cm are clipped before the log10 transform. The goal is to validate that descriptors developed from visible training data generalise without re-tuning against the held-out labels.
