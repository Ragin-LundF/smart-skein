# Single-label classification

One label per record, chosen by a softmax over competing labels. Use this when a record has exactly
one category. If two labels can be true at once, read [Multi-label](multi-label.md) instead.

## ClassificationService

Ties schema, vectorizer, classifier and storage together. **One service = one schema = one model.**

```kotlin
import io.skein.classify.application.ClassificationService
import io.skein.classify.domain.PrivacyModeEnum

val engine = ClassificationService(
    schema = schema,
    privacyMode = PrivacyModeEnum.FEATURES_ONLY,   // required, no default
    hashingConfig = config,                         // required
    // classifier   = NaiveBayesClassifier()        — default
    // featureStore = InMemoryFeatureStore()        — default
)

engine.learn(record)                    // one labelled record
engine.learnAll(trainingRecords)

val prediction = engine.classify(record = unlabelled)
prediction.label                        // the winner
prediction.confidence                   // its probability
prediction.alternatives                 // ranked, includes the winner

engine.feedback(record, correctLabel = Label(value = "AUTHENTICATION"))
engine.metrics()                        // totalObservations, perLabelCounts
engine.forget()                         // discards model and stored observations
```

### PrivacyModeEnum

| Mode | Meaning |
|---|---|
| `FEATURES_ONLY` | Only irreversible hashed features are stored |
| `ENCRYPTED_SOURCE` | The original record is also retained, encrypted at rest. Needs an encryption-capable store — see [`skein-store-postgres`](../store-postgres/README.md). With `InMemoryFeatureStore` both modes behave identically |

## Choosing a classifier

Both ship, both learn one observation at a time, both implement `Classifier`.

### Naive Bayes — the default

Multinomial Naive Bayes over hashed features with Laplace smoothing. A strong baseline that **learns
in a single pass**.

```kotlin
val classifier = NaiveBayesClassifier(smoothingAlpha = 1.0)
```

### Logistic regression (online SGD)

Discriminative, models correlated features better than Naive Bayes, but usually needs several passes
to converge.

```kotlin
val classifier = LogisticRegressionSgdClassifier(
    initialLearningRate = 0.1,
    decayRate = 0.0,           // lr(t) = lr0 / (1 + decayRate * step)
    l2Regularization = 0.0,
)

engine.learnAll(trainingRecords)
engine.retrain(epochs = 50, seed = 42)   // replay the stored corpus, shuffled
```

### Tuning

| Classifier | Parameter | Default | Move it |
|---|---|---|---|
| Naive Bayes | `smoothingAlpha` | 1.0 | Down for large confident data, up for sparse or noisy |
| Logistic regression | `initialLearningRate` | 0.1 | Down if the loss oscillates, up if convergence is slow |
| Logistic regression | `decayRate` | 0.0 | Above 0 to anneal over a long run |
| Logistic regression | `l2Regularization` | 0.0 | Above 0 to reduce overfitting on small data |

### retrain

```kotlin
fun retrain(epochs: Int = 1, seed: Long? = null)
```

Resets the classifier and replays every stored observation `epochs` times. The corpus is untouched,
so retraining is repeatable. `seed = null` replays in stored order (deterministic); a seed shuffles
each epoch deterministically, which is **recommended for SGD** — decorrelating consecutive updates
improves convergence. Naive Bayes is order-independent, so this mostly matters for SGD.

## Calibration and abstention

The raw softmax runs over unnormalised Naive Bayes log-likelihoods, which saturate near 0 and 1.
Those numbers are shown to people and drive uncertainty sampling, so the overconfidence is not
cosmetic.

```kotlin
engine.fitCalibration(heldOut = heldOutObservations)
engine.classify(record = record).confidence               // now calibrated

val prediction: Prediction? = engine.classifyOrNull(record = record, minConfidence = 0.85)
```

Temperature scaling is rank-preserving: it never changes which label wins, only how confident the
model claims to be. Accuracy and top-k are untouched while log loss and expected calibration error
improve.

**Fit on held-out data.** Scores on rows the model trained on already look calibrated, so fitting
there returns a temperature near 1 and silently does nothing.

## Explaining a prediction

```kotlin
val explanation = engine.explain(record = record, limit = 10)!!
explanation.contributions.forEach { contribution ->
    println("bucket ${contribution.featureIndex}  ${contribution.contribution}")
}
```

The decomposition is **exact**: `base` plus every feature's contribution equals `total`, and `total`
is the label's score minus the mean across labels — precisely the quantity the softmax turns into a
probability. No sampling, no surrogate model.

Contributions are mean-centred, which is what makes them readable. Every Naive Bayes term is a
log-probability and therefore negative, so ranking raw terms would surface whichever n-grams are
commonest overall. Centring turns each into a log-odds ratio against the average label, so a feature
equally likely under every label contributes exactly zero.

By default a contribution carries only its bucket id, which is safe to log. That is enough to spot a
single feature dominating a prediction — usually an identifier that leaked into the features — and to
watch influence drift between model versions.

```kotlin
engine.explain(record = record, mode = AttributionModeEnum.WITH_NGRAMS)
```

`WITH_NGRAMS` resolves buckets to source text. It builds no index and weakens no saved model, but
the result contains clear text — a per-call argument rather than a setting, so every site producing
clear text is greppable.

## Active learning

Do not label everything. Train on a little, ask the model which unlabelled records it is least sure
about, label those, feed them back.

```kotlin
import io.skein.classify.application.ActiveLearningSelector
import io.skein.classify.domain.UncertaintyStrategyEnum

val candidates = ActiveLearningSelector(service = engine).selectForReview(
    candidates = unlabelledRecords,
    limit = 10,
    strategy = UncertaintyStrategyEnum.MARGIN,
)

candidates.forEach { candidate ->
    // a human assigns the true label
    engine.feedback(candidate.record, correctLabel = humanLabel)
}
```

| Strategy | Measures | "Uncertain" means |
|---|---|---|
| `MARGIN` (default) | gap between the top two probabilities | two labels nearly tied |
| `LEAST_CONFIDENCE` | `1 − top probability` | low top probability |
| `ENTROPY` | Shannon entropy over all labels | a spread-out distribution |

Loop: `learnAll(seed)` → `selectForReview` → human labels → `feedback` → repeat until held-out
accuracy plateaus. Calibrate first — uncertainty sampling on uncalibrated Naive Bayes probabilities
picks badly.

## Storage

`ClassificationService` persists every observation to a `FeatureStore`, which is what `retrain`,
`metrics` and `forget` read and clear.

- `InMemoryFeatureStore` (default) — thread-safe, unbounded, lost on restart.
- `PostgresFeatureStore` — durable, with optional AES-256-GCM at rest. See
  [`skein-store-postgres`](../store-postgres/README.md).

The port is small (`add` / `addAll` / `all` / `labels` / `size` / `clear`); implement it to back the
corpus with anything.
