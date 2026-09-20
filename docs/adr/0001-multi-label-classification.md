# ADR 0001 — Multi-label classification in `skein-classify`

- **Status:** accepted
- **Date:** 2026-09-20
- **Affects:** `skein-classify` public API, `.skein` model format, `.ai/instructions/module-architecture.md`

## Context

`skein-classify`'s documented responsibility was *"assigning **one** label to a whole record"*, and
the `Classifier` port enforced it:

```kotlin
fun classify(features: FeatureVector): Prediction   // one winning label + softmax alternatives
```

A softmax makes labels compete: the scores are constrained to sum to one, which is the correct model
of a mutually exclusive choice and the wrong model of a taxonomy whose labels co-occur. A record
that is genuinely both `INSURANCE` and `LIFEINSURANCE` cannot be expressed, and a record that is
neither cannot be expressed either — the softmax always names a winner.

Supporting co-occurring labels needs three things the module did not have: independent per-label
scoring, a batch optimiser that converges (the existing classifier is online SGD), and a split that
keeps rule-derived siblings together. Everything else it already had, and had in a more general form
than a purpose-built alternative would: `Schema`, `HashingVectorizer`, `ModelStore`,
`ActiveLearningSelector`, calibration and explanation.

## Decision

### 1. Widen `skein-classify` rather than add a sibling module

The module's responsibility becomes *"assigning labels to a whole record — one label, or several
when labels co-occur"*, and `.ai/instructions/module-architecture.md` is updated to say so.

The alternative, a new published `skein-multilabel`, was rejected: it would have to depend on
`skein-classify` for `Schema`, `FeatureVector` and `HashingVectorizer` or duplicate them, and it
would split one capability — classification — across two published artifacts for no benefit the
consumer can see. The shared surface is large and the boundary would have been an artifact of how
the feature arrived rather than of what it is.

### 2. Two new ports, and `Classifier` untouched

`Classifier.classify` returning a single `Prediction` cannot express multi-label output, and the
repository's own rule is that *a published module's public API is a contract; prefer additive
change*. So:

```kotlin
interface MultiLabelClassifier {          // scoring, independent per label
    fun logits(features: FeatureVector): Map<Label, Double>
    fun scoreAll(features: FeatureVector): Map<Label, Double>   // independent sigmoids; do NOT sum to 1
    fun labels(): Set<Label>
}

interface BatchLearner {                  // fitting, needs the whole corpus
    fun fit(observations: List<MultiLabeledFeatures>): MultiLabelClassifier
}
```

`MultiLabelClassifier` and `Classifier` are siblings, and a caller picks one by asking a single
question: *can two labels be true at once?*

Fitting is a separate port because L-BFGS has no honest `learn(oneObservation)` — it needs the full
design matrix to compute a gradient at all. Widening `Classifier` would have forced a meaningless
method onto every implementation of it.

### 3. Feature hashing is kept

The prototype this capability came from used an explicit vocabulary plus IDF. That is not adopted.
Keyed feature hashing is load-bearing for the library's privacy positioning — there is deliberately
no default hashing key, *because it is a privacy decision* — and it is also the only option whose
fitting memory is constant in corpus size. An explicit vocabulary must hold every term it has seen
before it can apply a document-frequency floor, which is what made the prototype fail above a few
hundred thousand documents.

IDF over hashed buckets is available as an opt-in decorator (`IdfVectorizer`) for callers who
measure a benefit, with its fitted table persisted alongside the model.

### 4. A new payload shape in the `.skein` format, not a widened one

`ModelStore` persisted a corpus of observations and replayed it through a fresh classifier on load.
That works for classifiers that learn one observation at a time. It cannot express a batch-fitted
model: there is no incremental history to replay, "replaying" would mean re-running the optimiser
over the whole corpus on every file open, and the corpus would have to travel with the model to make
it possible at all.

So version `0x03` of the format selects a different payload — a fitted weight matrix — rather than
extending the observation payload. An older reader rejects a v3 file at the header instead of
misreading it, and `ModelStore.load` points a caller at `loadMultiLabel` by name.

### 5. Vectorizer identity is verified on load, fatally

`ModelStore.loadMultiLabel` takes the `Vectorizer` the caller intends to score with and throws
`VectorizerMismatchException` when its fingerprint differs from the one recorded at training time.

This is a hard failure rather than a warning because the failure it prevents is silent: a model
scored with a different featurisation still resolves every index and still multiplies every weight.
It returns confident, wrong labels, with no exception and nothing unusual in the output, for as long
as nobody notices.

## Consequences

**Gained.** Co-occurring labels, and the ability to return none. A batch optimiser that reaches the
true optimum. Honest scores on rule-derived corpora via `GroupedSplitter`. Weight persistence, which
also brings pruning and compact encoding into reach — together roughly a 10x reduction against an
unpruned float32 matrix at a measured cost of a handful of label decisions per million.

**Cost.** `skein-classify`'s surface is larger, and a reader must now choose between two ports.
`ClassifierKindEnum` gained an entry that `ClassifierFactory` cannot construct, which is an
asymmetry worth knowing about — it is rejected by name rather than silently substituted. The `.skein`
format has two payload shapes, and a v2-only reader cannot read a v3 file.

The `Vectorizer` port was what made `skein-classify-embedding-onnx` possible without
touching this module: it is a separate published artifact, so a consumer using feature hashing never
inherits an ONNX Runtime binary, and `skein-classify` gained no dependency at all.
