# skein-cli

Command-line tools for [`skein-classify`](../skein-classify). **Published.** Starts with one job:
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

A sample lives at [`src/test/resources/transactions.csv`](src/test/resources/transactions.csv) —
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

**`predict`**: `--input`, `--model`, `--out` (all required), `--epochs` (logreg rebuild, default `5`).
Adds `<label-col>` and `<label-col>_confidence` columns.

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

> Why no ONNX/external format? Skein is classical CPU ML with a keyed feature-hashing vectorizer that
> ONNX-ML can't represent, so an export couldn't reproduce features from raw text or be reloaded to
> keep training. The replay format above is dependency-free and round-trips losslessly.

## Build

```bash
./gradlew :skein-cli:test       # unit tests (CSV, model round-trip, labeling loop)
./gradlew :skein-cli:detekt     # static analysis
```
