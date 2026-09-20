# Featurisation

Everything between a record's text and the numbers a model sees. The default is keyed feature
hashing; the `Vectorizer` port lets you substitute anything else, including an
[embedding model](../embeddings/README.md).

## Feature hashing

Text becomes a sparse `FeatureVector` of **character n-grams (3–5)** and **word n-grams (1–2)**,
each hashed with **SipHash-2-4** — a keyed pseudo-random function — into a fixed feature space.

```kotlin
import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig

val config = HashingConfig(
    key0 = 0x1234_5678L,          // required, no default
    key1 = 0x9ABC_DEF0L,          // required, no default
    numFeatures = 1 shl 18,       // 262,144
    charNgramMin = 3, charNgramMax = 5,
    wordNgramMin = 1, wordNgramMax = 2,
)

val vectorizer = HashingVectorizer(config = config)
val features = vectorizer.vectorize(text = "login fails after update")
```

Character n-grams are what make this typo-tolerant: `insurance` and `insurnce` share most of their
3–5 character grams, so a misspelling still lands on mostly the same features.

## The hashing key is a privacy decision

`key0` and `key1` have **no default**, deliberately.

- A **fixed secret key** makes feature indices stable across runs and processes. This is required to
  persist or share a model. Keep it secret — it is what makes the hash a keyed PRF rather than a
  plain hash anyone can invert by brute force over a candidate vocabulary.
- `HashingConfig.randomKey()` gives a fresh `SecureRandom` key per process. Fine for a one-shot
  experiment; a model trained under one random key cannot be used under another.

### What the privacy guarantee is, exactly

The mapping is order-free and irreversible in aggregate: you cannot reconstruct the source text from
a feature vector. A stored model contains bucket ids and weights, not words.

Two honest limits:

- **Bucket ids are stable pseudonyms**, not noise. Under a fixed key the same n-gram always lands in
  the same bucket, which is what makes them useful for diagnosis — and means an adversary holding
  both the key and a candidate n-gram can confirm its bucket. That is why the key is secret.
- **`AttributionModeEnum.WITH_NGRAMS` returns clear text.** It re-derives n-grams from the record you
  pass in and stores nothing, so it grants nothing to someone who does not already hold the record.
  But the result *is* source text — do not log it where the record itself may not go.

The guarantee covers classification. The CRF tagger in [`skein-extract`](../extract/README.md) works
differently and can retain text fragments — see that module.

## Tuning

| Field | Default | Raising it |
|---|---|---|
| `numFeatures` | `2^18` | Fewer collisions, more memory and more model weights |
| `charNgramMin` / `Max` | 3 / 5 | Longer subword patterns, more features. Most typo tolerance lives here |
| `wordNgramMin` / `Max` | 1 / 2 | Bigrams capture short phrases, more features |
| `termWeighting` | `RAW_COUNT` | See below |

Pick `numFeatures` from measurement, not instinct. Hashing bounds the model — the width is fixed
whatever the corpus size — but it does not minimise it. At `2^20` a ~240-label model is around
151 MB against 37 MB at `2^18`, because a million buckets is far more than most data needs.

## Term weighting

```kotlin
HashingConfig(key0 = …, key1 = …, termWeighting = TermWeightingEnum.SUBLINEAR)
```

| Value | Feature value | Use when |
|---|---|---|
| `RAW_COUNT` (default) | occurrences | Records are similar in length |
| `SUBLINEAR` | `1 + ln(count)` | Record length varies a lot, or repeated boilerplate dominates |

It changes every feature value, so a model fitted under one weighting cannot be scored under the
other. The fingerprint covers it, which turns that from a silent wrong answer into a refused load.

## IDF over hashed buckets

Optional. Down-weights buckets that appear in nearly every record, so a feature earns influence by
being discriminating rather than common.

```kotlin
import io.skein.classify.application.IdfVectorizer

val idf = IdfVectorizer.fit(
    delegate = HashingVectorizer(config = config),
    texts = trainingTexts,            // training rows ONLY
)
```

Two things to get right:

- **Fit on the training rows of each fold**, never the whole corpus. The table is fitted state; a
  table built over everything carries held-out statistics into the training features, and every
  score afterwards is inflated. `MultiLabelCrossValidator` takes a vectorizer *factory* for exactly
  this reason.
- **The table must travel with the model.** `ModelStore.saveMultiLabel` writes it automatically when
  the vectorizer is an `IdfVectorizer`; read it back with the factory overload of `loadMultiLabel`.

The document-frequency floor scales with the corpus (`DocumentFrequencyTable.floorFor`): a bucket in
two documents out of seven thousand is weak evidence, and out of a million it is noise.

Treat this as a refinement to justify by measurement. On short records of similar length it often
buys very little.

## The `Vectorizer` port

```kotlin
interface Vectorizer {
    fun vectorize(text: String): FeatureVector
    fun dimension(): Int
    fun fingerprint(): VectorizerFingerprint
    fun canary(): VectorizerCanary? = null     // opt in; see Persistence
}
```

Implement it to feed features from anywhere. Everything downstream — the learners, the objective,
the scoring loop — sees only a `FeatureVector` and cannot tell the difference.

`fingerprint()` is the contract that keeps this safe: **two vectorizers sharing a fingerprint must
produce identical vectors for identical input.** It is written into the model file and verified on
load. Cover every setting that changes the output — for hashing that is the key, the width, the
n-gram ranges, the weighting and the normalizer; for an external model, the digest of the model file
itself.

Shipped implementations: `HashingVectorizer`, `IdfVectorizer`,
[`OnnxEmbeddingVectorizer`](../embeddings/onnx-local.md), and
[`HttpEmbeddingVectorizer`](../embeddings/external-service.md) in
`skein-classify-embedding-http`.

### Batching — the `BatchVectorizer` sub-port

```kotlin
interface BatchVectorizer : Vectorizer {
    fun vectorizeAll(texts: List<String>): List<FeatureVector>
}
```

Implement this **only** when featurising many texts at once is materially cheaper than one at a
time. For hashing it is not — a record costs microseconds and there is no per-call overhead worth
amortising — so `HashingVectorizer` deliberately stays a plain `Vectorizer`. For an encoder it is
one inference call instead of *n*, and for a service it is one HTTP round trip instead of *n*.
Both embedding adapters implement it.

Keeping it a separate type rather than a defaulted method is what makes the capability
*detectable*: code about to featurise a whole corpus can tell whether it is making one network
call or four hundred thousand.

Callers do not need the distinction. The extension picks the better path:

```kotlin
import io.skein.classify.spi.vectorizeAll

val vectors = vectorizer.vectorizeAll(texts = corpusTexts)   // batched if it can be, looped if not
```

Implementations must return **one vector per input, in input order**. A silently permuted batch
attaches every vector to the wrong record, which does not throw and does not look wrong.

`MultiLabelCrossValidator` uses this internally, which matters more than it sounds: it featurises
every row once *per fold*, so five-fold cross-validation over an embedding service was five round
trips per record before and is now two requests per fold.

One caveat: it returns every vector at once. Over a large corpus with an embedding vectorizer that
is real memory — chunk the call site if the corpus does not fit. `IdfVectorizer.fit` deliberately
still streams one text at a time for exactly this reason.

## Masking high-cardinality tokens

```kotlin
import io.skein.classify.application.TokenMasker

val masked = TokenMasker().mask(text = "order 998877665 shipped on 31.12.2024")
// "order <NUMERIC> shipped on <DATE>"
```

References, ids, dates and card numbers each become their own single-use feature otherwise. Masking
replaces them with their *type*, so a record's shape still distinguishes it from one with no
reference at all. Measured on a reference corpus: **59% fewer features and a 60% smaller model**, for
one tokenisation pass. See [Scale](scale.md).
