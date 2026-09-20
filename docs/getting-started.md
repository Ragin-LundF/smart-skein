# Getting started

A working multi-label classifier, from nothing. Roughly ten minutes.

## 1. Add the dependency

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-classify")
}
```

The BOM aligns every Skein module, so you never name a version twice. JDK 21 or newer.

## 2. Describe your records

```kotlin
import io.skein.classify.domain.Schema
import io.skein.classify.domain.SensitivityEnum

val schema = Schema.define {
    text(name = "subject")
    text(name = "body")
    categorical(name = "component")
    identifier(name = "reporterEmail", sensitivity = SensitivityEnum.PII)
    label(name = "tags")
}
```

A field marked `PII` never enters the feature text: it is not hashed, not stored in the model, and
not reachable from an explanation. Everything else is joined into the text the vectorizer sees.

## 3. Choose single- or multi-label

One question:

> **Can two labels be true of the same record at the same time?**

- **No** — use `Classifier`. Its softmax makes labels compete, which is the right model of a
  mutually exclusive choice. Continue in [single-label](classify/single-label.md).
- **Yes** — use `MultiLabelClassifier`. Each label gets an independent decision, so two can fire
  together and a record matching nothing comes back empty.

A support ticket can be a `BUG` *and* a `CRASH` *and* an `AUTHENTICATION` issue, so the rest of this
page is multi-label.

## 4. Train

```kotlin
import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner

val hashingConfig = HashingConfig(key0 = yourSecretKey0, key1 = yourSecretKey1)
val vectorizer = HashingVectorizer(config = hashingConfig)

val corpus = listOf(
    "login fails after update" to setOf("BUG", "AUTHENTICATION"),
    "app closes when opening settings" to setOf("BUG", "CRASH"),
    "how do I reset my password" to setOf("DOCUMENTATION", "AUTHENTICATION"),
    "crash on startup, stack trace attached" to setOf("BUG", "CRASH"),
    "typo in the onboarding guide" to setOf("DOCUMENTATION"),
)

val model = LbfgsMultiLabelLearner(featureCount = vectorizer.dimension()).fit(
    observations = corpus.map { (text, tags) ->
        MultiLabeledFeatures(
            features = vectorizer.vectorize(text = text),
            labels = tags.map { tag -> Label(value = tag) }.toSet(),
        )
    },
)
```

`key0` and `key1` have no default. The hashing key is the secret that makes feature indices
irreversible, and choosing it is a privacy decision the library will not make for you. Use a fixed,
secret key whenever a model is persisted or shared — a random one makes yesterday's model
unreadable.

## 5. Score a record

```kotlin
val prediction = model.predict(
    features = vectorizer.vectorize(text = "the app crashes when I sign in"),
    threshold = 0.5,
)

println(prediction.labels())              // e.g. [BUG, CRASH, AUTHENTICATION]
println(prediction.topK(count = 3))       // ranked, with probabilities
```

The probabilities are independent sigmoids. They do **not** sum to one, and reading them as a
distribution is the most common way to misuse multi-label output.

## 6. Find out whether it is any good

```kotlin
import io.skein.classify.application.MultiLabelCrossValidator
import io.skein.classify.application.MultiLabelEvaluator
import io.skein.classify.domain.MultiLabeledText

val report = MultiLabelCrossValidator().crossValidate(
    corpus = corpus.map { (text, tags) ->
        MultiLabeledText(
            featureText = text,
            labels = tags.map { tag -> Label(value = tag) }.toSet(),
            group = null,
        )
    },
    vectorizerFactory = { HashingVectorizer(config = hashingConfig) },
    learnerFactory = { LbfgsMultiLabelLearner(featureCount = vectorizer.dimension()) },
    folds = 5,
)

MultiLabelEvaluator().sweep(outcomes = report.pooledOutcomes).forEach { point ->
    println("${point.threshold}  P ${point.precision}  R ${point.recall}  covered ${point.coverage}")
}
```

Two things to internalise before trusting any number here:

- **If your labels came from a rule engine, set `group`** to whatever produced the row. Rows sharing
  an origin share their wording, so a random split puts siblings on both sides and the score
  measures memorisation. See [Evaluation](classify/evaluation.md).
- **There is no correct threshold.** The sweep is a table because the trade between a wrong label
  and a missing one is a business question.

## 7. Save and load

```kotlin
import io.skein.classify.application.ModelStore
import io.skein.classify.infrastructure.MultiLabelLogisticClassifier

ModelStore.saveMultiLabel(
    path = Path.of("tickets.skein"),
    schema = schema,
    model = model as MultiLabelLogisticClassifier,
    vectorizer = vectorizer,
    hashingConfig = hashingConfig,
)

val loaded = ModelStore.loadMultiLabel(path = Path.of("tickets.skein"), vectorizer = vectorizer)
```

`loadMultiLabel` throws when the vectorizer differs from the one the model was trained with. That is
deliberate: a model scored with a different featurisation does not fail, it returns confident wrong
labels.

## Where to go next

| You want to | Read |
|---|---|
| Understand the whole workflow, including where labels come from | [Bring your own data](bring-your-own-data.md) |
| Tune features, or understand the privacy guarantee | [Featurisation](classify/featurisation.md) |
| Measure honestly on rule-derived labels | [Evaluation](classify/evaluation.md) |
| Match meaning rather than literal wording | [Embeddings](embeddings/README.md) |
| Train on more than a million records | [Scale](classify/scale.md) |
| Run something immediately | [`examples`](../examples) |
