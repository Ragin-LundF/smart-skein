# skein-classify

Assigning labels to a whole record — one label, or several when labels co-occur.

Classical statistical machine learning on the CPU: no GPU, no neural network, no external service.
Features are irreversible keyed hashes by default, so personal data never enters a model in clear
text. Depends on `skein-text`.

> **Audience:** developers integrating classification, and data scientists who need to know how
> features are built, which model is doing the work, how to train and tune it, and how to keep a
> human in the loop.

## Use it when

- Records need a category: a ticket, a transaction, a document, a log line.
- Wording is messy — typos, abbreviations, inconsistent formatting.
- Labels must be auditable: every prediction decomposes exactly into the features that drove it.
- Personal data must not end up in a model file.
- You are replacing a rule engine and want the model to learn from the rules' output.

## The one decision

> **Can two labels be true of the same record at the same time?**

**No** → `Classifier`. A softmax makes labels compete, which is correct for a mutually exclusive
choice, and it learns one record at a time so incremental and active learning work.

**Yes** → `MultiLabelClassifier` with `BatchLearner`. Independent per-label decisions, so several
can fire and a record matching nothing comes back empty.

Forcing co-occurring labels through a softmax suppresses the second label by construction.

## At a glance

```kotlin
import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner

val vectorizer = HashingVectorizer(config = HashingConfig(key0 = secret0, key1 = secret1))

val model = LbfgsMultiLabelLearner(featureCount = vectorizer.dimension()).fit(
    observations = corpus.map { row ->
        MultiLabeledFeatures(
            features = vectorizer.vectorize(text = row.text),
            labels = row.tags.map { tag -> Label(value = tag) }.toSet(),
        )
    },
)

model.predict(features = vectorizer.vectorize(text = "login fails after update"), threshold = 0.5)
    .labels()   // e.g. [BUG, AUTHENTICATION]
```

`key0` and `key1` have no default. The hashing key is what makes feature indices irreversible, and
choosing it is a privacy decision the library will not make for you.

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-classify")
}
```

## Documentation

| | |
|---|---|
| [Overview](../docs/classify/README.md) | The module, the single/multi-label decision, package layout |
| [Schema and records](../docs/classify/schema.md) | Field types, keeping PII out of features, bulk import |
| [Featurisation](../docs/classify/featurisation.md) | Hashing, n-grams, the privacy guarantee, IDF, the `Vectorizer` port |
| [Single-label](../docs/classify/single-label.md) | Naive Bayes, online SGD, calibration, explanations, active learning |
| [Multi-label](../docs/classify/multi-label.md) | L-BFGS batch training, per-label thresholds |
| [Evaluation](../docs/classify/evaluation.md) | Metrics, grouped cross-validation, threshold sweeps |
| [Persistence](../docs/classify/persistence.md) | The `.skein` format, fingerprints, model size |
| [Scale](../docs/classify/scale.md) | Above a million records |

Semantic features instead of literal n-grams:
[`skein-classify-embedding-onnx`](../skein-classify-embedding-onnx) and
[Embeddings](../docs/embeddings/README.md).

New here? [Getting started](../docs/getting-started.md) ·
[Bring your own data](../docs/bring-your-own-data.md)
