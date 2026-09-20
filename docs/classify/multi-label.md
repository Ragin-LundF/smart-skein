# Multi-label classification

Any number of labels per record, including none. Each label gets its own independent one-vs-rest
decision, so `BUG` and `CRASH` and `AUTHENTICATION` can all fire on one ticket — and a record
matching nothing comes back empty.

Use this when two labels can be true at once. If a record has exactly one category, use
[single-label](single-label.md): a softmax is the correct model of a mutually exclusive choice.

## Why a softmax is wrong here

A softmax constrains its scores to sum to one. That is exactly right when the labels compete and
exactly wrong when they co-occur: a genuine second label is suppressed by construction, and no
threshold or tuning recovers it. There is also no way to say "none of these" — a softmax always
names a winner.

Independent sigmoids have neither problem. The price is that the scores are **not** a distribution:
two labels at 0.9 is ordinary, and reading them as probabilities that sum to one is the most common
way to misuse the output.

## Two ports

```kotlin
interface MultiLabelClassifier {
    fun logits(features: FeatureVector): Map<Label, Double>
    fun scoreAll(features: FeatureVector): Map<Label, Double>   // independent sigmoids
    fun predict(features: FeatureVector, threshold: Double): MultiLabelPrediction
    fun labels(): Set<Label>
    fun explain(features: FeatureVector, label: Label, limit: Int): Explanation?
}

interface BatchLearner {
    fun fit(observations: List<MultiLabeledFeatures>): MultiLabelClassifier
}
```

Fitting is a separate port because the optimiser needs the whole corpus at once. L-BFGS has no
meaningful `learn(oneObservation)` — it needs the full design matrix to compute a gradient at all —
so widening `Classifier` would have forced a meaningless method onto every implementation.

The practical consequence: **there is no incremental update.** New data means refitting. In exchange
you get an optimiser that reaches the true optimum rather than wandering near it.

## Training

```kotlin
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner

val learner = LbfgsMultiLabelLearner(
    featureCount = vectorizer.dimension(),
    inverseRegularization = 10.0,   // C, on scikit-learn's scale
    gradientTolerance = 1e-4,
    maxIterations = 500,
    historySize = 5,
    keepFraction = 0.08,
)

val model = learner.fit(
    observations = corpus.map { row ->
        MultiLabeledFeatures(
            features = vectorizer.vectorize(text = row.text),
            labels = row.tags.map { tag -> Label(value = tag) }.toSet(),
            group = row.originatingRule,     // see Evaluation
        )
    },
)
```

`featureCount` must be the vectorizer's full feature space, not the widest index this corpus happens
to use. Deriving it from the data would give each cross-validation fold a different feature space,
and a model fitted on one fold could not score a record from another.

### Empty label sets are meaningful

```kotlin
MultiLabeledFeatures(features = …, labels = emptySet())
```

An explicit negative says "none of the known labels apply here", which is what teaches each
one-vs-rest head where its label does *not* belong. A corpus of only positives cannot express that,
and models trained on one tend to over-fire.

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `inverseRegularization` (C) | 10.0 | Larger means less regularisation. Same meaning as scikit-learn's `C` |
| `gradientTolerance` | 1e-4 | Convergence test on the largest gradient component |
| `maxIterations` | 500 | Per label |
| `historySize` | 5 | Curvature pairs retained. Drives memory: `historySize × 2 × featureCount` doubles **per worker** |
| `keepFraction` | 0.08 | Share of weights kept by magnitude. See [Persistence](persistence.md) |

Labels are fitted in parallel, bounded by `TrainingParallelism` from available heap and feature
count rather than core count. At a wide feature space, parallelism is how a training run exhausts
the heap — see [Scale](scale.md).

## Predicting

```kotlin
val prediction = model.predict(features = vectorizer.vectorize(text = text), threshold = 0.5)

prediction.labels()                 // Set<Label> at or above the threshold
prediction.topK(count = 3)          // ranked, threshold ignored
prediction.probabilityOf(label)
prediction.at(threshold = 0.8)      // re-read the same scores, no rescoring
```

`at()` costs nothing, which is what makes a threshold sweep free — see [Evaluation](evaluation.md).

### Prefer topK for human review

Measured on a reference corpus: `recall@1 = 0.484` against `recall@3 = 0.775`. A reviewer shown
three candidates is right far more often than one shown only the winner. If a person is in the loop,
`topK` is usually the right shape, not a threshold.

## Per-label thresholds

One global threshold assumes every label's scores are calibrated the same way. They are not: a label
with four hundred examples produces confident, well-spread probabilities, while one with four
produces a timid cluster that never reaches 0.5. A global cut silences the rare label entirely while
the well-supported one over-fires at the same value.

```kotlin
import io.skein.classify.application.ThresholdOptimizer

val thresholds = ThresholdOptimizer().fit(outcomes = report.pooledOutcomes)

val accepted = thresholds.accepted(prediction = prediction)
thresholds.of(label = Label(value = "CRASH"))
```

Each label's threshold is chosen to maximise that label's own F1, by an exact sweep over every
distinct score. **Fit on held-out data** — cross-validated outcomes are ideal, because every row
there was scored by a model that had not seen it. Thresholds fitted on training rows sit exactly
where that model's training scores happen to fall.

`LabelThresholds` persists with the model.

## Explaining

```kotlin
val explanation = model.explain(features = features, label = Label(value = "CRASH"), limit = 10)!!
explanation.base          // the intercept
explanation.total         // the logit
explanation.contributions // ranked by absolute magnitude
```

Unlike the single-label explanation, nothing is centred across labels: each head is an independent
linear model, so `intercept + Σ(weight × value)` *is* the logit. The decomposition is exact rather
than a shift-invariant stand-in.

## What is not available

| | Why |
|---|---|
| Incremental learning | L-BFGS needs the whole corpus. Refit instead |
| Active learning | `ActiveLearningSelector` works against `Classifier`. Uncertainty over independent sigmoids needs a different definition |
| Temperature calibration | `Calibration` is softmax-specific: it minimises a log-sum-exp likelihood against a single true label. Per-label Platt scaling is the multi-label equivalent; `ThresholdOptimizer` covers the common need |
| `ConfusionMatrix` | Does not apply. With co-occurring labels there is no single predicted class to confuse with a single true class |
