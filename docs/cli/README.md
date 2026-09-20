# skein-cli

Command-line tools for [`skein-classify`](../../skein-classify). **Published.** Starts with one job:
**help you turn a pile of records into labeled training data**, with the model doing the boring part.

> **Audience:** anyone with a CSV of records and a column they want predicted. You label the few rows
> the model is least sure about; it learns from each answer and gets better as you go (active learning).

The tool is published as a normal library artifact **and** as a runnable application, so you can run
it standalone without ever adding it to your own project's runtime classpath — see
[Enable it locally](install.md).

## Pages

| | |
|---|---|
| [Flags](flags.md) | Every flag on every command, and the CSV contract |
| [Install](install.md) | Running the CLI locally without polluting a runtime |
| [Evaluate](evaluate.md) | `evaluate`, the metrics it reports, and scaling to large pools |
| [Predict](predict.md) | Abstention, and what a saved model contains |

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

## Build

```bash
./gradlew :skein-cli:test       # unit tests (CSV, model round-trip, labeling loop)
./gradlew :skein-cli:detekt     # static analysis
```

---

[Flags](flags.md) · [Install](install.md) · [Evaluate](evaluate.md) · [Predict](predict.md) · [All documentation](../README.md)
