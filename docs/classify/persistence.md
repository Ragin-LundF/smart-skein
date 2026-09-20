# Persistence

Models are written to a `.skein` file: a 4-byte magic `SKEI`, a version byte, then a GZIP-compressed
ProtoBuf payload.

The version byte selects the **payload shape**, not merely a revision.

| Version | Payload | Written by |
|---|---|---|
| `0x01`, `0x02` | A corpus of observations, replayed through a fresh classifier on load | `ModelStore.save` |
| `0x03` | An already-fitted weight matrix | `ModelStore.saveMultiLabel` |

## Why two shapes

Single-label classifiers learn one observation at a time, so persisting the observations and
replaying them reconstructs the model exactly — and keeps the corpus available for `retrain`.

A batch-fitted multi-label model has no such history. Replaying would mean re-running L-BFGS over
the whole corpus on every file open, and the corpus would have to travel with the model to make it
possible at all. So v3 stores the fitted weights.

An older reader rejects a v3 file at the header rather than misreading it, and `ModelStore.load`
points you at `loadMultiLabel` by name.

## Single-label

```kotlin
ModelStore.save(
    path = path,
    schema = engine.schema,
    classifier = ClassifierKindEnum.LOGISTIC_REGRESSION,
    hashingConfig = hashingConfig,
    observations = engine.featureStore.all(),
    calibration = engine.calibration,
    hyperparameters = engine.classifier.hyperparameters(),   // without this, defaults come back
)

val loaded = ModelStore.load(path = path)
val classifier = ClassifierFactory.create(
    kind = loaded.classifier,
    hyperparameters = loaded.hyperparameters,
)
```

`save` receives the classifier *kind*, not the instance, so it cannot discover the tuning by itself.
Pass `hyperparameters` or a tuned model comes back as a default-constructed one — a different model
that looks like the right one.

## Multi-label

```kotlin
ModelStore.saveMultiLabel(
    path = path,
    schema = schema,
    model = model as MultiLabelLogisticClassifier,
    vectorizer = vectorizer,
    thresholds = thresholds,          // optional, defaults to uniform 0.5
    hashingConfig = hashingConfig,    // optional, for inspection
)

val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer)
loaded.classifier
loaded.thresholds
loaded.fingerprint
```

A fitted `IdfVectorizer` writes its document-frequency table into the file automatically, because it
cannot be reconstructed from anything the caller holds. Read it back with the factory overload:

```kotlin
val loaded = ModelStore.loadMultiLabel(
    path = path,
    vectorizerFactory = { table -> IdfVectorizer(delegate = base, table = requireNotNull(table)) },
)
```

## The fingerprint check

`loadMultiLabel` takes the vectorizer you intend to score with and throws
`VectorizerMismatchException` when its fingerprint differs from the one recorded at training time.

This is fatal rather than a warning because the failure it prevents is silent. A model scored with a
different featurisation still resolves every index and multiplies every weight. It returns confident
wrong labels, with no exception and nothing unusual in the output, for as long as nobody notices.

For hashing, the fingerprint covers the key, the width, both n-gram ranges, the term weighting and
the normalizer's identity. For an [ONNX embedding](../embeddings/onnx-local.md) it covers the digest
of the model file's **contents** — a model swapped in place under an unchanged name is exactly what
this catches.

## How big the file is

Size obeys one formula, and the corpus is not in it:

```
bytes ≈ features × labels × keepFraction × bytesPerWeight
```

Measured on a 237-label, 102,205-feature model at 8% density: **89.7% of the file is the weight
matrix.** Every vocabulary string together would be 5.6%, which is why stopword removal cannot be
the reason a model is large.

Three levers, and they multiply:

### Prune harder

L2 shrinks weights towards zero but never to it, so a fitted matrix is dense and mostly noise.

| `keepFraction` | Size | Records with identical label sets | Differing decisions |
|---|---|---|---|
| 1.000 | 195.6 MB | — | — |
| 0.080 | 17.3 MB | 99.986% | 1 in 1.7 M |
| **0.040** | **9.5 MB** | **99.904%** | **7 in 1.7 M** |
| 0.020 | 5.7 MB | 98.03% | 159 |
| 0.010 | 3.7 MB | 78.5% | 1,821 |

The 0.08 default is conservative by 2–4×. The cliff is between 2% and 1% and is sharp — locate it on
your own data rather than assuming this transfers.

The cutoff is global, not per label, which tracks how much each label was actually learned but means
a rare label can be pruned into silence. Check macro-F1 after changing it.

### Compact encoding — free

Applied automatically. Weights are stored as `float16` where the range allows, with an automatic
fallback to `float32`; label indices are delta-coded per feature row, which makes the gaps small
enough for the container's variable-length integers. Measured worst-case confidence drift from
`float16`: 0.000424, with **zero** changed label decisions.

### Fewer features

Masking cut features by 59% and the model by 60%. See [Scale](scale.md).

## Privacy note

A `.skein` file contains the hashing key and the feature vectors. The features are irreversible, but
the key is the secret that makes them so — **treat the file as a secret**.
