# skein-cli

Command-line tools for [`skein-classify`](../../skein-classify). **Published.** Starts with one job:
**help you turn a pile of records into labeled training data**, with the model doing the boring part.

> **Audience:** anyone with a CSV of records and a column they want predicted. You label the few rows
> the model is least sure about; it learns from each answer and gets better as you go (active learning).

The tool is published as a normal library artifact **and** as a runnable application, so you can run
it standalone without ever adding it to your own project's runtime classpath — see
[Enable it locally](#enable-it-locally-without-runtime-pollution).

## Commands

| Command | What it does |
|---------|--------------|
| `label`   | Train on the rows that already have a label, then repeatedly surface the **most-uncertain** unlabeled rows for you to confirm or correct. Writes the enriched CSV and saves the model. |
| `predict` | Load a saved model and classify every row of an input CSV (predicted label + confidence). |
| `export`  | Convert a binary `.skein` model file to a human-readable text file for inspection. |
| `evaluate`| Measure model quality — accuracy, per-class precision/recall/F1, a confusion matrix, top-k, log loss and calibration — with an optional `--min-accuracy` gate for CI. |

## Run it

```bash
# Active-learning labeling loop over a CSV with a "category" column:
./gradlew :skein-cli:run --args="label \
  --input transactions.csv --label-col category \
  --out labeled.csv --model model.skein --budget 20"
```

Or unzip the published distribution and use the launcher script:

```bash
./gradlew :skein-cli:installDist
skein-cli/build/install/skein-cli/bin/skein-cli label --input transactions.csv --label-col category --out labeled.csv --model model.skein
```

For each uncertain row you get the model's current guess and can accept it, type the right label,
skip, or quit:

```
──────────────────────────────
  purpose: Allianz-Life 42.50 insurance premium
  iban: DE18
  suggested: insurance (confidence 0.71)
  ranked: insurance 0.71, salary 0.20, rent 0.09
  label [Enter=accept, <text>=set, s=skip, q=quit]:
```

At the end it writes the now-labeled CSV, saves the model, and prints what it learned.

### Tutorial data

A sample lives at [`src/test/resources/transactions.csv`](../../skein-cli/src/test/resources/transactions.csv) —
bank-transaction purposes with `insurance` / `rent` / `salary` categories (the last few rows are
deliberately unlabeled so you can try the loop):

```bash
./gradlew :skein-cli:run --args="label \
  --input skein-cli/src/test/resources/transactions.csv --label-col category \
  --out /tmp/labeled.csv --model /tmp/model.skein"
```

Then reuse the saved model to classify everything in one shot:

```bash
./gradlew :skein-cli:run --args="predict \
  --input skein-cli/src/test/resources/transactions.csv --model /tmp/model.skein --out /tmp/pred.csv"
```

## Flags

**`label`**

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--input <csv>` | yes | — | Input records (first row is the header). |
| `--label-col <name>` | yes | — | The column holding the label (empty cells = "to be labeled"). |
| `--out <csv>` | yes | — | Where to write the enriched, now-labeled records. |
| `--model <file>` | no | — | Model file: **loaded if it exists** (resume), and saved at the end. Omit it and the model is not persisted. |
| `--classifier nb\|logreg` | no | `nb` | Classifier for a fresh model. Ignored when resuming (the file carries it). |
| `--budget <n>` | no | `20` | Max rows to label this run. |
| `--batch <n>` | no | `8` | How many candidates are surfaced before re-ranking by uncertainty. |
| `--strategy margin\|least-confidence\|entropy` | no | `margin` | How uncertainty is measured. |
| `--epochs <n>` | no | `5` | SGD passes when training/rebuilding a `logreg` model. |
| `--key <k0>,<k1>` | no | random | Fixed hashing key for a fresh model (see [Persistence](#persistence--privacy)). |
| `--scan-limit <n>` | no | `0` | Rows scored per round; `0` scans the whole pool (exact). See [Scaling](#scaling-to-large-pools). |

**`predict`**: `--input`, `--model`, `--out` (all required), `--epochs` (logreg rebuild, default `5`).
Adds `<label-col>` and `<label-col>_confidence` columns.

**`export`**

| Flag | Required | Meaning |
|------|----------|---------|
| `--model <file>` | yes | Source `.skein` model file. |
| `--out <file>` | yes | Destination text file. |

Writes a plain-text representation of the model — schema fields, classifier type, hashing config,
and all labeled observations as sparse feature vectors:

```
skein-model 1
classifier NAIVE_BAYES
hashing <key0> <key1> 262144 3 5 1 2
field TEXT purpose PUBLIC
field IDENTIFIER iban PUBLIC
field LABEL category
---
insurance	1:0.5 3:0.8 15:1.0
rent	2:0.3 7:1.0
```

Useful for auditing what a model learned, diffing two model versions, or just understanding the
schema. The hashing key is present in the output — treat the exported file with the same care as the
binary model (see [Persistence & privacy](#persistence--privacy)).

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

## Enable it locally without runtime pollution

Most consumers depend on `skein-classify` for the library and only need the CLI as a *tool*. Don't put
it on your `implementation`/`runtimeClasspath` — declare it in an **isolated configuration** and run it
via a `JavaExec` task. It stays out of your application jar entirely:

```kotlin
val skeinCli by configurations.creating

dependencies {
    skeinCli("io.github.ragin-lundf:skein-cli:<version>")
}

tasks.register<JavaExec>("skeinLabel") {
    classpath = configurations["skeinCli"]   // isolated — never in your runtime classpath
    mainClass = "io.skein.cli.MainKt"
    args(
        "label",
        "--input", "data.csv",
        "--label-col", "category",
        "--out", "labeled.csv",
        "--model", "model.skein",
    )
}
```

Then `./gradlew skeinLabel`. Nothing from `skein-cli` leaks into your shipped artifact.

## `evaluate` — is the model any good?

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

---

## Abstaining in `predict`

```bash
skein predict --model model.skein --input in.csv --out out.csv --min-confidence 0.85
```

A row below the threshold is written with an **empty** label cell and its confidence intact. That is
deliberate: `skein label` treats a blank label as pending, so piping the output back into `label`
hand-labels exactly the rows the model refused.

> `predict` overwrites the label column unconditionally, so any ground-truth label in the input is
> replaced — and an abstained row blanks that cell. Keep your gold labels in a separate file.
>
> Thresholds are only meaningful against a **calibrated** model. Raw Naive Bayes confidences sit
> near 0 and 1, so no threshold between them separates anything.

---

## Persistence & privacy

The model file is a compact text file holding the schema, the **fixed hashing key**, the classifier
kind, and every labeled feature vector. On reload the model is rebuilt by *replaying* those
observations (`ClassificationService.retrain`), so Naive Bayes is reproduced exactly and logistic
regression deterministically — there is nothing classifier-internal to go stale.

- **A fresh run generates a random key and saves it** in the model file, so resuming and `predict`
  reproduce identical feature indices. Pass `--key <k0>,<k1>` if you want to fix it yourself.
- The file stores only the irreversible **hashed** feature vectors and labels — never clear-text record
  content (the engine runs in `FEATURES_ONLY` privacy mode). **But it contains the hashing key**, which
  is the privacy secret, so treat the model file as sensitive and don't commit it.

## Build

```bash
./gradlew :skein-cli:test       # unit tests (CSV, model round-trip, labeling loop)
./gradlew :skein-cli:detekt     # static analysis
```
