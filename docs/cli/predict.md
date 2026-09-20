# Abstaining in `predict`


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

---

[Command line](README.md) · [Flags](flags.md) · [Install](install.md) · [Evaluate](evaluate.md) · [Predict](predict.md) · [All documentation](../README.md)
