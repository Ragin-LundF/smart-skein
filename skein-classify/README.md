# skein-classify

Record → label classification: typo-tolerant, self-training, privacy-preserving.
Depends on `skein-text`.

## Foundation (this phase)

- **Schema** — `Schema.define { … }` DSL over a sealed `FieldSpec` hierarchy
  (`TextField`, `CategoricalField`, `NumericField`, `IdentifierField`, `LabelField`) with a
  `SensitivityEnum` (`PUBLIC`/`PII`).
- **Validation & inference** — `SchemaValidator` (rejects missing labels / non-numeric numbers,
  warns on missing/unknown fields) and `SchemaInference` (numeric / categorical / text heuristics).
- **Feature hashing** — `HashingVectorizer` turns text into a sparse `FeatureVector` of char+word
  n-grams, each keyed-hashed with **SipHash-2-4** into a fixed feature space. Order-free,
  irreversible. PII fields never reach the feature text (`RecordMapper`).
- **Import pipeline** — `RecordSource` (streaming) → `RecordImportService` →
  `ImportResult` (`accepted` / `rejected` / `warnings`).
- **Storage** — `FeatureStore` SPI + `InMemoryFeatureStore`.

## Quick start

```kotlin
val schema = Schema.define {
    text("purpose")
    numeric("amount")
    identifier("iban")     // PII — excluded from features by default
    label("category")
}

val result = RecordImportService(schema).importFrom(csvRecordSource)
val vectorizer = HashingVectorizer()                 // production: supply a secret HashingConfig key
val features = vectorizer.vectorize(result.accepted.first().featureText)
```

## Package layout

```
io.skein.classify
├─ domain/          Record, Schema + FieldSpec hierarchy, SensitivityEnum, Label,
│                   FeatureVector, LabeledFeatures, HashingConfig, ValidationResult, ImportResult …
├─ application/     SchemaValidator, SchemaInference, HashingVectorizer, RecordMapper, RecordImportService
├─ spi/             FeatureStore, RecordSource (ports)
└─ infrastructure/  SipHash, InMemoryFeatureStore
```

> **Privacy:** `HashingConfig.key0/key1` is the keyed-hash secret. Defaults are deterministic for
> dev/tests; **production must supply its own secret key**.

## Learning (Phase 4)

- **`Classifier`** SPI port + incremental **`NaiveBayesClassifier`** (multinomial NB over hashed
  features, Laplace smoothing, stable-softmax probabilities).
- **`ClassificationService`** — `learn` / `learnAll` / `classify` / `feedback` / `metrics` / `forget`,
  driving the classifier and `FeatureStore` from mapped, vectorized records.
- **`PrivacyModeEnum`** (`FEATURES_ONLY` / `ENCRYPTED_SOURCE`) — required, no default.
- **`Prediction`** — winning `label`, `confidence`, and full ranked `alternatives` (`ScoredLabel`).

```kotlin
val engine = ClassificationService(schema, privacyMode = PrivacyModeEnum.FEATURES_ONLY)
engine.learnAll(trainingRecords)
val prediction = engine.classify(record)          // prediction.label, prediction.confidence
engine.feedback(record, correctLabel = Label("housing"))
```

## Self-training (Phase 5)

- **`LogisticRegressionSgdClassifier`** — discriminative "v2" `Classifier`: multinomial logistic
  regression with online SGD, sparse per-label weights, optional L2. Shares probability calibration
  with Naive Bayes via `PredictionFactory` (stable softmax).
- **`ClassificationService.retrain(epochs)`** — batch retraining: replays stored observations N
  times (deterministic), useful for SGD models. Keeps the stored corpus intact.
- **`ActiveLearningSelector`** — surfaces the least-certain records (margin sampling) as
  `ReviewCandidate`s for human labeling.

```kotlin
val engine = ClassificationService(schema, PrivacyModeEnum.FEATURES_ONLY, classifier = LogisticRegressionSgdClassifier())
engine.learnAll(trainingRecords)
engine.retrain(epochs = 50)                                   // multi-pass SGD from the store
val toLabel = ActiveLearningSelector(engine).selectForReview(unlabeled, limit = 10)
```

## Coming next (Phase 6)

`skein-extract`: typed pattern DSL, slot definitions, pattern matching, template clustering.
