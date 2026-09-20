# ADR 0001 — Multi-label classification in `skein-classify`

- **Status:** accepted
- **Date:** 2026-09-20
- **Affects:** `skein-classify` public API, `.skein` model format, `.ai/instructions/module-architecture.md`

## Context

A softmax makes labels compete. Its scores are constrained to sum to one, which is the correct model
of a mutually exclusive choice and the wrong model of a taxonomy whose labels co-occur. Under a
softmax a record that is genuinely both `INSURANCE` and `LIFEINSURANCE` cannot be expressed, and
neither can a record that is nothing at all — the softmax always names a winner.

Supporting co-occurring labels takes three things a single-label classifier does not need:
independent per-label scoring, a batch optimiser that converges, and a split that keeps
rule-derived siblings together. Everything else `skein-classify` already provides, and provides in
a more general form than a purpose-built alternative would: `Schema`, `HashingVectorizer`,
`ModelStore`, `ActiveLearningSelector`, calibration and explanation.

## Decision

### 1. `skein-classify` covers both, rather than a sibling module covering one

The module's responsibility is *"assigning labels to a whole record — one label, or several when
labels co-occur"*, and `.ai/instructions/module-architecture.md` says so.

A separate published `skein-multilabel` is the rejected alternative. It would have to depend on
`skein-classify` for `Schema`, `FeatureVector` and `HashingVectorizer` or duplicate them, and it
would split one capability across two artifacts for no benefit a consumer can see. The shared
surface is large, and the boundary would mark how the feature arrived rather than what it is.

### 2. Two ports for multi-label, and `Classifier` untouched

`Classifier.classify` returns a single `Prediction` and cannot express multi-label output. A
published module's public API is a contract, and additive change is preferred, so multi-label gets
its own ports instead of a widened one:

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

`MultiLabelClassifier` and `Classifier` are siblings, not a ranking. A caller picks one by asking a
single question: *can two labels be true at once?*

Fitting is a separate port because L-BFGS has no honest `learn(oneObservation)` — it needs the full
design matrix to compute a gradient at all. Folding it into `Classifier` would force a meaningless
method onto every implementation.

### 3. Feature hashing, not an explicit vocabulary

Keyed feature hashing is load-bearing for the library's privacy positioning — there is deliberately
no default hashing key, *because choosing one is a privacy decision* — and it is the only option
whose fitting memory is constant in corpus size. An explicit vocabulary has to hold every term it
has seen before it can apply a document-frequency floor, which is what puts a ceiling of a few
hundred thousand documents on that approach.

IDF over hashed buckets is available as an opt-in decorator (`IdfVectorizer`) for callers who
measure a benefit, with its fitted table persisted alongside the model.

### 4. A separate payload shape in the `.skein` format, not a widened one

The single-label format persists a corpus of observations and replays it through a fresh classifier
on load. That works for classifiers that learn one observation at a time. It cannot express a
batch-fitted model: there is no incremental history to replay, "replaying" would mean re-running the
optimiser over the whole corpus on every file open, and the corpus would have to travel with the
model to make it possible at all.

So version `0x03` of the format selects a different payload — a fitted weight matrix — rather than
extending the observation payload. The version byte names the payload *shape*, so a reader that
does not know v3 rejects the file at the header instead of misreading it, and `ModelStore.load`
points a caller at `loadMultiLabel` by name.

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

**Cost.** `skein-classify`'s surface is larger, and a reader has to choose between two ports.
`ClassifierKindEnum` carries an entry that `ClassifierFactory` cannot construct, which is an
asymmetry worth knowing about — it is rejected by name rather than silently substituted. The
`.skein` format has two payload shapes, and a v2-only reader cannot read a v3 file.

**Knock-on.** The `Vectorizer` port is what lets an embedding adapter supply semantic features
without this module changing. `BatchVectorizer` extends it for the implementations where
featurising a corpus in one call is materially cheaper than a loop — the same "a port for the
implementations that can honestly offer it" reasoning that separates `BatchLearner` from
`Classifier` above. Both adapters —
[`skein-classify-embedding-onnx`](../embeddings/onnx-local.md) and
[`skein-classify-embedding-http`](../embeddings/external-service.md) — are separate published
artifacts, so a consumer using feature hashing inherits neither an ONNX Runtime binary nor a JSON
parser, and `skein-classify` depends on neither. The identity guarantee in §5 is only as strong as
what a vectorizer can prove about itself, which is what [ADR 0002](0002-vector-canary.md) addresses.
