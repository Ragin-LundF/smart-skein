# skein-classify

Record → label classification: **typo-tolerant, self-training, privacy-preserving.** Assigns one
label to a whole record (a bank transaction, a support ticket, a document) using classical
statistical ML — no neural networks, no GPU, no external services. Depends on `skein-text`.

> **Audience:** developers integrating classification, and data scientists who need to know exactly
> how features are built, which model is doing the work, how to **train / retrain / tune** it, and
> how to keep a human in the loop via active learning.

## Why classical ML here

- **Incremental & online** — every model learns one record at a time; you can `learn` forever, no
  full retrain needed (though batch `retrain` is available for SGD models).
- **Privacy by construction** — text becomes an irreversible, keyed feature hash before any model
  sees it; PII fields never enter the feature text at all.
- **Cheap & explainable** — runs on a CPU, predictions come with full ranked alternatives.

## Installation

```kotlin
dependencies {
    implementation("io.skein:skein-classify:<version>")     // align via skein-bom
}
```

---

## The pipeline at a glance

```
Record (field map)
   │  RecordMapper        — drops PII + label, joins PUBLIC fields → featureText
   ▼
featureText (String)
   │  HashingVectorizer   — char + word n-grams → SipHash → sparse FeatureVector
   ▼
FeatureVector
   │  Classifier          — NaiveBayes (default) or LogisticRegressionSGD
   ▼
Prediction (label, confidence, ranked alternatives)
```

`ClassificationService` ties all of this together. **One service = one schema = one model.**

---

## 1. Define a schema

A schema declares your fields and their sensitivity. It needs **exactly one `label`** (the target).

```kotlin
val schema = Schema.define {
    text("purpose")                 // free text → contributes to features
    categorical("counterparty")     // low-cardinality category → features
    numeric("amount")               // numeric → features (and validated as numeric)
    identifier("iban")              // identifier → defaults to PII, excluded from features
    label("category")               // the target label (exactly one required)
}
```

### Field types (`FieldSpec`)

| Builder | Type | Default sensitivity | In feature text? |
|---------|------|---------------------|------------------|
| `text(name)` | `TextField` | `PUBLIC` | yes |
| `categorical(name)` | `CategoricalField` | `PUBLIC` | yes |
| `numeric(name)` | `NumericField` | `PUBLIC` | yes (also validated numeric) |
| `identifier(name)` | `IdentifierField` | **`PII`** | **no** (PII excluded) |
| `label(name)` | `LabelField` | `PUBLIC` (fixed) | no (it's the target) |

Every builder except `label` takes an optional `sensitivity` override:

```kotlin
text("notes", sensitivity = SensitivityEnum.PII)        // force a text field out of features
identifier("public_ref", sensitivity = SensitivityEnum.PUBLIC)  // let an id into features
```

> **Privacy rule:** a field reaches the feature text **only** if it is not the label and its
> sensitivity is not `PII`. `RecordMapper` enforces this — PII never becomes a feature.

### Don't have a schema? Infer one

`SchemaInference` proposes a schema from sample records using simple heuristics:

```kotlin
val schema = SchemaInference(maxCategoricalCardinality = 20).infer(records, labelField = "category")
```

Per non-label field, looking at non-null sample values:

1. all parse as numbers → `NumericField`
2. values repeat **and** distinct count ≤ `maxCategoricalCardinality` (default 20) → `CategoricalField`
3. all values distinct **and** look like codes (alphanumeric, ≥4 chars, contains a digit) → `IdentifierField` (PII)
4. otherwise → `TextField`

Always **review an inferred schema** — heuristics guess; you know your domain.

### Validate records — `SchemaValidator`

```kotlin
val result = SchemaValidator(schema).validate(record)
result.isValid()    // false if there are errors
result.errors       // e.g. "label is missing", "amount is not numeric"  → record rejected
result.warnings     // e.g. "field 'memo' not in schema"                 → record still accepted
```

---

## 2. Feature hashing — `HashingVectorizer` + `HashingConfig`

Text becomes a sparse `FeatureVector` of **character n-grams (3–5)** and **word n-grams (1–2)**, each
hashed with **SipHash-2-4** (a keyed PRF) into a fixed feature space. The mapping is order-free and
irreversible in aggregate — you cannot reconstruct the source text from the vector.

```kotlin
val config = HashingConfig(
    key0 = 0x1234_5678L,   // REQUIRED — the keyed-hash secret (no default; it's a privacy choice)
    key1 = 0x9ABC_DEF0L,   // REQUIRED
    // numFeatures   = 262_144 (2^18)   — feature space size
    // charNgramMin  = 3, charNgramMax = 5
    // wordNgramMin  = 1, wordNgramMax = 2
)
val vectorizer = HashingVectorizer(config = config)
val features = vectorizer.vectorize("aig-life 67,89 insurance premium")
features.nonZeroCount()
```

> ⚠️ **`key0`/`key1` have no default — you must choose them.** This is deliberate:
> - **Fixed secret key** → feature indices are stable across runs/processes → required to **persist
>   or share a trained model**. Keep the key secret; it's what makes the hash a keyed PRF.
> - `HashingConfig.randomKey()` → a fresh `SecureRandom` key per process. Convenient for one-shot
>   demos, but a model trained under one random key **cannot** be reused under another.

### Tuning knobs (data scientists)

| Field | Default | Effect of raising |
|-------|---------|-------------------|
| `numFeatures` | `2^18 = 262 144` | fewer hash collisions, more memory/weights |
| `charNgramMin / Max` | `3 / 5` | captures longer subword patterns; more features (typo robustness comes largely from char n-grams) |
| `wordNgramMin / Max` | `1 / 2` | bigrams capture short phrases; more features |

Char n-grams are what make classification **typo-tolerant**: `"insurance"` and `"insurnce"` still
share most of their 3–5-char grams.

---

## 3. Train and classify — `ClassificationService`

```kotlin
val engine = ClassificationService(
    schema = schema,
    privacyMode = PrivacyModeEnum.FEATURES_ONLY,    // REQUIRED (no default)
    hashingConfig = config,                          // REQUIRED
    // classifier   = NaiveBayesClassifier()         — default model
    // featureStore = InMemoryFeatureStore()         — default storage
)

// --- TRAIN ---
engine.learn(record)                                 // one labeled record
engine.learnAll(trainingRecords)                     // many

// --- PREDICT ---
val prediction = engine.classify(unlabeledRecord)
prediction.label          // winning Label
prediction.confidence     // probability of the winner, 0.0..1.0
prediction.alternatives   // List<ScoredLabel> ranked high→low (includes the winner)

// --- CORRECT (online feedback) ---
engine.feedback(record, correctLabel = Label("housing"))   // learns the record under the right label

// --- INSPECT ---
val m = engine.metrics()
m.totalObservations       // Int
m.perLabelCounts          // Map<Label, Int> — class balance

// --- RESET ---
engine.forget()           // discards model AND stored observations
```

### Records are just maps

```kotlin
val record = Record(values = mapOf(
    "purpose" to "AIG-Life 67,89 insurance premium",
    "iban"    to "DE00...",          // PII — present, but excluded from features
    "category" to "insurance",        // the label (omit it for prediction)
))
```

For prediction the label field can be absent. For `learn` it must be present (else
`IllegalArgumentException`).

### `PrivacyModeEnum` (required, no default)

| Mode | Meaning |
|------|---------|
| `FEATURES_ONLY` | Only the irreversible hashed features are stored. |
| `ENCRYPTED_SOURCE` | The original record is also retained, **encrypted at rest**. Requires an encryption-capable store — see `skein-store-postgres`. With the default `InMemoryFeatureStore`, both modes behave identically (features only). |

---

## 4. Choosing and tuning the model

Two classifiers ship, both **incremental** and sharing the same softmax probability calibration
(`PredictionFactory`, numerically stable). Both implement the `Classifier` SPI, so you can plug in
your own.

### Naive Bayes (default) — `NaiveBayesClassifier`

Generative multinomial NB over hashed features with Laplace smoothing. **Strong baseline, learns in
a single pass**, no retraining needed.

```kotlin
val nb = NaiveBayesClassifier(smoothingAlpha = 1.0)   // default α = 1.0 (Laplace)
val engine = ClassificationService(schema, PrivacyModeEnum.FEATURES_ONLY, config, classifier = nb)
engine.learnAll(trainingRecords)                       // one pass is enough
```

- `smoothingAlpha` — additive smoothing over the feature vocabulary. Lower (→ 0) trusts the data
  more (riskier on sparse classes); higher smooths harder. Default `1.0` is a safe start.

### Logistic Regression (SGD) — `LogisticRegressionSgdClassifier`

Discriminative multinomial logistic regression trained online with SGD. Models correlated features
better than NB, but **usually needs multiple passes** to converge — use `retrain(epochs)`.

```kotlin
val lr = LogisticRegressionSgdClassifier(
    initialLearningRate = 0.1,    // default
    decayRate = 0.0,              // default: constant LR;  lr(t) = lr0 / (1 + decayRate * step)
    l2Regularization = 0.0,       // default: no L2;  raise to fight overfitting
)
val engine = ClassificationService(schema, PrivacyModeEnum.FEATURES_ONLY, config, classifier = lr)

engine.learnAll(trainingRecords)
engine.retrain(epochs = 50)       // replay the stored corpus 50× (see below)
```

### Hyperparameter cheat-sheet (data scientists)

| Classifier | Param | Default | Tune toward |
|------------|-------|---------|-------------|
| NaiveBayes | `smoothingAlpha` | `1.0` | ↓ for confident large data, ↑ for sparse/noisy |
| LogisticRegression | `initialLearningRate` | `0.1` | ↓ if loss oscillates, ↑ if convergence is slow |
| LogisticRegression | `decayRate` | `0.0` | `>0` to anneal LR over a long training run |
| LogisticRegression | `l2Regularization` | `0.0` | `>0` to reduce overfitting on small data |

### `retrain` — batch passes from the stored corpus

```kotlin
fun retrain(epochs: Int = 1, seed: Long? = null)
```

Resets the classifier and replays **every stored observation** `epochs` times. The stored corpus is
kept intact (only the model state is rebuilt), so retraining is repeatable.

- `seed == null` (default) → replays in stored order: fully deterministic and resumable.
- `seed != null` → shuffles each epoch deterministically. **Recommended for SGD**: decorrelating
  consecutive updates improves convergence.

```kotlin
engine.retrain(epochs = 50, seed = 42)    // 50 shuffled passes, reproducible
```

Naive Bayes is order-independent, so `retrain` mostly matters for the SGD model.

---

## 5. Keep a human in the loop — `ActiveLearningSelector`

Don't label everything. Train on a little, then ask the model which **unlabeled** records it's most
unsure about, label those, and feed them back. This is the cheapest path to accuracy.

```kotlin
val selector = ActiveLearningSelector(engine)
val candidates = selector.selectForReview(
    candidates = unlabeledRecords,
    limit = 10,
    strategy = UncertaintyStrategyEnum.MARGIN,   // default
)

candidates.forEach { c ->
    println("guess=${c.prediction.label.value} margin=${c.margin}")
    // a human assigns the true label, then:
    engine.feedback(c.record, correctLabel = humanLabel)
}
```

Returns up to `limit` `ReviewCandidate`s (`record`, model `prediction`, uncertainty `margin`),
**most-uncertain first**.

### Uncertainty strategies — `UncertaintyStrategyEnum`

| Strategy | Measures | "Uncertain" means |
|----------|----------|-------------------|
| `MARGIN` (default) | gap between top-1 and top-2 probabilities | small gap (two labels nearly tied) |
| `LEAST_CONFIDENCE` | `1 − top probability` | low top probability |
| `ENTROPY` | Shannon entropy over all label probabilities | spread-out distribution |

A typical loop: `learnAll(seed)` → `selectForReview` → human labels → `feedback` → repeat until
`metrics()` and held-out accuracy plateau.

---

## 6. Storage — `FeatureStore`

`ClassificationService` persists every observation (label + feature vector) to a `FeatureStore`,
which is what `retrain`, `metrics`, and `forget` read/clear.

- `InMemoryFeatureStore` (default) — thread-safe, unbounded, lost on restart.
- `PostgresFeatureStore` (`skein-store-postgres`) — durable, with optional AES-256-GCM at rest for
  `ENCRYPTED_SOURCE`. Plug it in via the `featureStore` constructor param.

```kotlin
val engine = ClassificationService(
    schema, PrivacyModeEnum.ENCRYPTED_SOURCE, config,
    featureStore = postgresStore,        // see skein-store-postgres
)
```

The `FeatureStore` SPI (`add` / `addAll` / `all` / `labels` / `size` / `clear`) is small — implement
it to back the corpus with anything you like.

---

## 7. Bulk import — `RecordImportService`

For ingesting a stream (CSV, DB cursor, …) with validation in one pass:

```kotlin
val source = object : RecordSource {                 // streaming, constant memory
    override fun stream(): Sequence<Record> = csvRows.map { Record(it) }
}
val result = RecordImportService(schema).importFrom(source)

result.accepted        // List<MappedRecord> — passed validation
result.rejected        // List<RejectedRecord> — with per-record reasons
result.warnings        // non-fatal notes (unknown/missing fields)
result.acceptedCount()
```

`MappedRecord` is `(featureText, label?)` — the already-mapped, PII-stripped view ready to learn from.

---

## Worked example: train from scratch

```kotlin
val schema = Schema.define {
    text("purpose"); identifier("iban"); label("category")
}
val config = HashingConfig(key0 = 1L, key1 = 2L)      // fixed key → model is reproducible
val engine = ClassificationService(schema, PrivacyModeEnum.FEATURES_ONLY, config)

engine.learnAll(listOf(
    Record(mapOf("purpose" to "AIG-Life 67,89 insurance premium", "iban" to "DE00", "category" to "insurance")),
    Record(mapOf("purpose" to "Allstate-Home 90,00 insurance premium", "iban" to "DE01", "category" to "insurance")),
    Record(mapOf("purpose" to "rent apartment monthly", "iban" to "DE02", "category" to "rent")),
    Record(mapOf("purpose" to "salary october payout", "iban" to "DE03", "category" to "salary")),
))

val p = engine.classify(Record(mapOf("purpose" to "Geico-Auto 120,00 insurance premium", "iban" to "DE99")))
println("${p.label.value} @ ${"%.2f".format(p.confidence)}")   // insurance @ 1.00
```

See [`examples`](../examples) for the full **classify → route → extract** pipeline that combines this
module with `skein-extract`.

---

## Package layout

```
io.skein.classify
├─ domain/          Record, Schema + FieldSpec hierarchy, SensitivityEnum, Label, FeatureVector,
│                   LabeledFeatures, HashingConfig, Prediction/ScoredLabel, PredictionFactory,
│                   PrivacyModeEnum, UncertaintyStrategyEnum, ReviewCandidate, ClassificationMetrics,
│                   ImportResult/RejectedRecord/MappedRecord, ValidationResult
├─ application/     SchemaValidator, SchemaInference, HashingVectorizer, RecordMapper,
│                   RecordImportService, ClassificationService, ActiveLearningSelector
├─ spi/             Classifier, FeatureStore, RecordSource (ports)
└─ infrastructure/  SipHash, InMemoryFeatureStore, NaiveBayesClassifier, LogisticRegressionSgdClassifier
```

> **Privacy summary:** PII fields never enter `featureText`; feature text is hashed with a keyed,
> irreversible PRF; `key0/key1` are a required secret you control; and `ENCRYPTED_SOURCE` + an
> encrypting store is the only way original content is retained, always encrypted at rest.

