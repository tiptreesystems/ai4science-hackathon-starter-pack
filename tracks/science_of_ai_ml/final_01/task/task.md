# Fair Tabular Binary Classification

Build a binary classifier for tabular records that include a protected-group
attribute. The training file contains labels; the test file has the same
feature schema without labels.

Write one binary decision for every test row. The hidden scorer computes
absolute average-odds difference from the held-out labels and protected-group
metadata. Lower disparity is better, but the primary score also applies an
accuracy utility guard so all-constant low-utility submissions are not
competitive.

Your `run.sh` must write `predictions.csv`:

```csv
id,prediction
test_0001,0
test_0002,1
```

The file must contain exactly these two columns, `id` and `prediction`.
`prediction` must be a binary value, `0` or `1`.

Use only the provided task files and public documentation for modeling and
method development. Do not join the rows against external tabular datasets or
external labels.
