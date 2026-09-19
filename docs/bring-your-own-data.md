# Bring your own data

Six steps from a pile of your records to a model you can defend. Each one is a few lines of real
code, and each one has a decision in it that is yours rather than the library's.

## 1. Describe your records

A `Schema` declares what a record holds. Field types are not decoration: they decide what becomes a
feature and what never leaves your process.

```kotlin
val schema = Schema.define {
    text(name = "title")
    text(name = "body")
    categorical(name = "channel")
    numeric(name = "minutes")
    identifier(name = "accountIban", sensitivity = SensitivityEnum.PII)
    label(name = "tags")
}
```

A field marked `SensitivityEnum.PII` is excluded from the feature text entirely — it is never
hashed, never stored in a model, and never reaches a log line. Everything else is concatenated by
`RecordMapper` into the text the vectorizer sees.

## 2. Get the data in

Implement `RecordSource` for wherever your records live, or hand an existing sequence to
`RecordImportService`, which validates against the schema and reports what it rejected rather than
silently dropping it.

If you do not know your schema yet, `SchemaInference` proposes one from a sample. Read what it
proposes — it guesses types from values, and it cannot guess which of your fields are personal data.

## 3. Decide where labels come from

Three routes, and the honest answer is usually a mix:

| Route | When it fits | What it costs |
|---|---|---|
| **Distil an existing rule engine** | You already have rules in production | Labels inherit the rules' blind spots — see step 5 |
| **Label by hand** | No rules, and the taxonomy is small | Time, and it is the only route that produces genuinely independent truth |
| **`ActiveLearningSelector`** | A large unlabelled pile and some budget | Picks the records whose labels teach the most, so a fixed review budget goes further |

Distillation is the fastest start and the one with a trap in it. A model trained on rule-generated
labels largely learns the rules; it does not learn what the rules were trying to express. Step 5 is
how you find out which you got.

## 4. Choose single- or multi-label

One question decides it:

> **Can two labels be true of the same record at the same time?**

**No** — the record has exactly one category. Use `Classifier`. Its softmax makes the labels compete,
which is the correct model of a mutually exclusive choice.

**Yes** — labels co-occur, or none may apply. Use `MultiLabelClassifier`. Every label gets its own
independent decision, so two can fire together and a record that matches nothing comes back empty.

Getting this wrong is not a rounding error. Forcing co-occurring labels through a softmax
constrains the scores to sum to one, so a genuine second label is suppressed by construction, and no
amount of tuning recovers it.

```kotlin
val vectorizer = HashingVectorizer(config = HashingConfig(key0 = yourKey0, key1 = yourKey1))
val learner = LbfgsMultiLabelLearner(featureCount = vectorizer.dimension())

val model = learner.fit(
    observations = corpus.map { row ->
        MultiLabeledFeatures(
            features = vectorizer.vectorize(text = row.featureText),
            labels = row.labels,
            group = row.group,
        )
    },
)
```

`key0`/`key1` have no default on purpose: the hashing key is the secret that makes feature indices
irreversible, and choosing it is a privacy decision the library will not make for you. Use a fixed,
secret key whenever a model must be persisted or shared.

## 5. Train and evaluate — and group the split if labels came from rules

```kotlin
val report = MultiLabelCrossValidator().crossValidate(
    corpus = corpus,                                   // MultiLabeledText, with `group` set
    vectorizerFactory = { HashingVectorizer(config = hashingConfig) },
    learnerFactory = { LbfgsMultiLabelLearner(featureCount = featureCount) },
    folds = 5,
)
```

Set `group` to whatever produced the row — the rule, the template, the source document. Rows sharing
an origin share their wording, so a random split puts siblings on both sides and the model reads the
answer off the training set. On real rule-distilled data that difference measured **micro-F1 0.79
random against 0.75 grouped, and macro-F1 0.634 against 0.534**. The grouped number is the honest
one, and grouping costs nothing when it turns out not to be needed.

Two further habits worth keeping:

- **Watch macro-F1, not only micro.** Micro pools every label decision, so frequent labels dominate
  it. Macro weights a label with four examples the same as one with four hundred, and on a
  long-tailed taxonomy it is always the lower and more informative number. Per-label accuracy tracks
  examples-per-label almost exactly — measured macro-F1 **0.099** at 1–4 examples per label against
  **0.784** at 100+ — so a hybrid that keeps rules for the sparse tail is a design, not a compromise.
- **Keep a hand-labelled held-out set**, separate from anything a rule produced, using wording the
  rules do not cover. It is the only measurement that answers whether the model generalises past the
  rules, and it will score worse. That gap is the finding.

## 6. Choose a threshold

```kotlin
MultiLabelEvaluator().sweep(outcomes = report.pooledOutcomes).forEach { point ->
    println("${point.threshold}  P ${point.precision}  R ${point.recall}  covered ${point.coverage}")
}
```

There is no correct row in that table, and the library will not pick one. Accepting more labels
always trades precision for recall; which side is right depends on what a wrong label costs against
what a missing one costs, and that is a business question.

Read the **coverage** column while you do it. A threshold with excellent precision that labels a
third of your records is a different product from one that labels all of them, and precision alone
hides the difference.

Two refinements, once you have the table:

- **`topK(n)` instead of a threshold**, when a person reviews the output. Correct labels sit in the
  top 3 far more often than they win outright — measured recall@1 0.484 against recall@3 0.775 — so
  a reviewer shown three candidates is right far more often than one shown the winner.
- **`ThresholdOptimizer`** fits a threshold per label. A rare label's scores cluster low and a global
  cut silences it entirely while a well-supported label over-fires at the same cut. Fit it on
  held-out data — cross-validated outcomes are ideal — never on the rows the model trained on.

## Saving and loading

```kotlin
ModelStore.saveMultiLabel(
    path = path,
    schema = schema,
    model = model as MultiLabelLogisticClassifier,
    vectorizer = vectorizer,
    thresholds = thresholds,
    hashingConfig = hashingConfig,
)

val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer)
```

`loadMultiLabel` **throws** when the vectorizer you pass is not the one the model was trained with.
That is deliberate and it is not configurable: a model scored with a different featurisation does
not fail — it returns confident, wrong labels, silently, for as long as nobody notices.

## At scale

Four things matter above a few hundred thousand records, in this order:

1. **Mask high-cardinality tokens** with `TokenMasker` before featurising. References, ids, dates and
   card numbers each become their own single-use feature otherwise. Measured: **59% fewer features
   and a 60% smaller model**, for one tokenisation pass.
2. **Deduplicate** with `CorpusDeduplicator`, on masked text *and* label set. Once the references are
   masked away, a large share of a real export is literally the same record — measured 400,000
   documents collapsing to 21,420. Report the ratio on your own data rather than assuming that one
   transfers.
3. **Prune**, via `LbfgsMultiLabelLearner`'s `keepFraction`. L2 shrinks weights towards zero but
   never to it, so a fitted matrix is dense and mostly noise. Choose the fraction from a measured
   size/accuracy sweep on your data; the 8% default is deliberately conservative.
4. **Let the learner bound its own parallelism.** It already does, from heap and feature count rather
   than core count — each concurrent fit holds its own optimiser history, so at a wide feature space
   parallelism is how a training run exhausts the heap.

## A worked example

`examples/` contains `recipes`, which runs the whole of the above on a tagging problem where labels
genuinely co-occur — a dish is `VEGETARIAN` **and** `ITALIAN` **and** `QUICK`. It prints the grouped
and ungrouped cross-validation side by side, sweeps the threshold, and scores a hand-written
held-out set using wording the rules never mention, which is the number that matters.

```bash
./gradlew :examples:run --args="recipes"
```
