# Flags


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
| `--key <k0>,<k1>` | no | random | Fixed hashing key for a fresh model (see [Persistence](predict.md#persistence--privacy)). |
| `--scan-limit <n>` | no | `0` | Rows scored per round; `0` scans the whole pool (exact). See [Scaling](evaluate.md#scaling-to-large-pools). |

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
binary model (see [Persistence & privacy](predict.md#persistence--privacy)).

---

[Command line](README.md) · [Flags](flags.md) · [Install](install.md) · [Evaluate](evaluate.md) · [Predict](predict.md) · [All documentation](../README.md)
