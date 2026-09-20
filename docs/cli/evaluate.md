# `evaluate` — is the model any good?


```bash
skein evaluate --model model.skein --input labeled.csv          # score the SAVED model
skein evaluate --model model.skein --folds 5                    # cross-validate the recipe
skein evaluate --model model.skein --min-accuracy 0.80          # CI gate; exits 2 on failure
```

| Flag | Meaning |
|---|---|
| `--model <file>` | saved `.skein` model (required) |
| `--input <csv>` | labeled CSV to score the saved model against. Mutually exclusive with `--folds`/`--test-ratio` |
| `--folds <n>` | stratified k-fold over the stored observations, retraining per fold (n ≥ 2) |
| `--test-ratio <r>` | stratified holdout over the stored observations (default 0.2) |
| `--top-k <n>` | rank depth for top-k accuracy (default 3) |
| `--bins <n>` | calibration bins (default 10) |
| `--seed <n>` | split seed (default 42) |
| `--epochs <n>` | training passes over a split or fold (default 5) |
| `--out <file>` | also write the text report here |
| `--csv <file>` | per-label metrics as CSV |
| `--confusion <file>` | the full confusion matrix as CSV |
| `--min-accuracy <r>` | exit 2 when accuracy falls below r |
| `--delimiter <char>` | CSV field delimiter |

A `.skein` file stores the **training** observations, so `--folds` and `--test-ratio` measure the
recipe by retraining — not the saved model, which already saw every stored row. Only `--input`
measures the saved model. The report header always says which mode ran, and there is deliberately no
way to score a model against its own training rows.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | report produced; the quality gate passed or was not requested |
| `1` | error — unknown or missing flag, missing model, unreadable or label-less input |
| `2` | report produced, but `--min-accuracy` was not met |

## Scaling to large pools

Active learning shines when the unlabeled pool is huge — public datasets, or data generated from
rules — and you only hand-label the uncertain few. The selection loop is built for that:

- **Vectorize once.** Each row's hashed feature vector (the expensive part) is computed on first
  sight and cached, so re-ranking after every answer re-hashes nothing.
- **Parallel scoring.** Vectorizing and scoring fan out across all CPU cores.
- **Bounded top-K.** Candidates are ranked with a size-`batch` heap, not a full sort of the pool.
- **`--scan-limit`.** With `0` (default) every round scores the whole pool — exact, and fast into the
  low millions thanks to the above. For very large pools, set `--scan-limit 100000` (say) and each
  round scores only a random window of that many rows, making per-round cost independent of pool size
  and capping memory (only sampled rows are vectorized/cached).

Reference point — a 1,000,000-row pool labeling 20 rows on a laptop: ~4s exact (`--scan-limit 0`),
~1s sampled (`--scan-limit 20000`).

**Remaining ceiling:** all rows are read into memory at once (the CSV reader materializes the file).
For pools beyond available heap, pre-split the input or use `--scan-limit` with a sharded input.

---

[Command line](README.md) · [Flags](flags.md) · [Install](install.md) · [Evaluate](evaluate.md) · [Predict](predict.md) · [All documentation](../README.md)
