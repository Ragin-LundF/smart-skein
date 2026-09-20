# Route A — your own embedding model, in-process

The model runs inside your JVM through ONNX Runtime. No network, no service to operate, and the
model file's bytes are hashed into the fingerprint so a swapped model is refused rather than
silently producing wrong labels.

This is the route to take for anything you run in production.

## 1. Add the module

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-classify")
    implementation("io.github.ragin-lundf:skein-classify-embedding-onnx")
}
```

This module carries the ONNX Runtime and tokenizer native binaries. It is separate from
`skein-classify` precisely so that consumers using feature hashing never inherit them.

## 2. Export a model to ONNX

A one-off Python step. Hugging Face Optimum handles the conversion:

```bash
pip install "optimum[exporters]"

optimum-cli export onnx \
  --model intfloat/multilingual-e5-small \
  --task feature-extraction \
  e5-small-onnx/
```

That writes `e5-small-onnx/model.onnx` and `e5-small-onnx/tokenizer.json`. Those two files are all
Skein needs — copy them wherever your application can read them.

`--task feature-extraction` is the important flag: it exports the encoder with `last_hidden_state`
as the output, which is the per-token tensor this adapter pools. An export for a different task
produces a different output and will not work.

### Making it smaller

```bash
optimum-cli export onnx \
  --model intfloat/multilingual-e5-small \
  --task feature-extraction \
  --optimize O2 \
  e5-small-onnx/
```

Dynamic int8 quantisation roughly quarters the file and speeds up CPU inference substantially, at a
small accuracy cost. Quantise *before* training, not after — a quantised model produces different
vectors, so a classifier trained on the fp32 vectors is invalidated by the switch. The fingerprint
enforces that, which is the behaviour you want.

## 3. Use it

```kotlin
import io.skein.classify.embedding.onnx.application.EmbeddingCache
import io.skein.classify.embedding.onnx.domain.NormalizationEnum
import io.skein.classify.embedding.onnx.domain.PoolingStrategyEnum
import io.skein.classify.embedding.onnx.infrastructure.OnnxEmbeddingVectorizer
import java.nio.file.Path

OnnxEmbeddingVectorizer.open(
    modelPath = Path.of("e5-small-onnx/model.onnx"),
    tokenizerPath = Path.of("e5-small-onnx/tokenizer.json"),
    pooling = PoolingStrategyEnum.MEAN,
    normalization = NormalizationEnum.L2,
    cache = EmbeddingCache(),
).use { vectorizer ->
    val features = vectorizer.vectorizeAll(texts = corpusTexts)
    // ... train, evaluate, save
}
```

It holds native resources, so close it — `use` is the easy way.

### The E5 prefix

E5-family models are trained with `query: ` and `passage: ` markers. For classification every record
is a passage, so prepend `passage: ` consistently — on the training corpus **and** at inference:

```kotlin
val features = vectorizer.vectorizeAll(texts = corpusTexts.map { text -> "passage: $text" })
```

Omitting it does not fail. It costs accuracy, quietly. Models outside the E5 family generally want
no prefix; check the card.

> If you standardise on a prefix, wrap the vectorizer so it cannot be forgotten at inference time.
> A small decorator implementing `Vectorizer` and delegating after prepending is enough, and it
> keeps the prefix inside the object whose fingerprint covers it.

## 4. Train and save

Nothing here is embedding-specific — a dense vector travels through the same sparse `FeatureVector`
as hashed n-grams.

```kotlin
import io.skein.classify.application.ModelStore
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner
import io.skein.classify.infrastructure.MultiLabelLogisticClassifier

val model = LbfgsMultiLabelLearner(
    featureCount = vectorizer.dimension(),
    keepFraction = 1.0,
).fit(
    observations = corpus.mapIndexed { index, row ->
        MultiLabeledFeatures(features = features[index], labels = row.labels)
    },
)

ModelStore.saveMultiLabel(
    path = Path.of("tickets.skein"),
    schema = schema,
    model = model as MultiLabelLogisticClassifier,
    vectorizer = vectorizer,
)
```

`keepFraction = 1.0` because pruning is pointless on a dense model: 384 × 237 × 4 bytes is 364 KB,
and every weight carries signal. Pruning exists for sparse n-gram matrices two orders of magnitude
larger.

## 5. Load at inference

```kotlin
OnnxEmbeddingVectorizer.open(
    modelPath = Path.of("e5-small-onnx/model.onnx"),
    tokenizerPath = Path.of("e5-small-onnx/tokenizer.json"),
).use { vectorizer ->
    val loaded = ModelStore.loadMultiLabel(path = Path.of("tickets.skein"), vectorizer = vectorizer)

    val prediction = loaded.classifier.predict(
        features = vectorizer.vectorize(text = "passage: the app crashes when I sign in"),
        threshold = loaded.thresholds.fallback,
    )
}
```

If the model file, the tokenizer, the pooling strategy or the normalisation differs from training,
this throws `VectorizerMismatchException` instead of returning a model. Deploy the `.skein` file and
the two model files together and this never fires; when it does, it has caught a real problem.

## Deployment

Three artifacts travel together: `model.onnx`, `tokenizer.json`, and your `.skein` file. Ship them
as a unit — a container layer, an archive, an object-store prefix. Versioning them independently is
how the mismatch happens in the first place.

| Concern | What to do |
|---|---|
| Size | The ONNX model dominates. Quantise, and pick 384 dimensions over 768. |
| Startup | Session creation reads the whole model. Open the vectorizer once and keep it. |
| Memory | Roughly the model file, plus working tensors proportional to batch size × sequence length. |
| Threads | The vectorizer is safe to share. ONNX Runtime parallelises inside a call, so serve batches rather than racing single records. |
| Warm-up | The first inference is slower. Embed one throwaway record at startup if you have a latency SLO. |

## Caching

```kotlin
cache = EmbeddingCache(maxEntries = 100_000)
```

Keyed by fingerprint plus a digest of the text, so it cannot serve a vector computed under a
different configuration. It earns its keep whenever the same text is embedded twice:

- **Hyperparameter sweeps** re-embed the whole corpus per candidate. With a cache, once.
- **Cross-validation** re-embeds each fold's training rows. With a cache, once.
- **Real corpora repeat.** After masking, a large share of records are identical — see
  [Scale](../classify/scale.md). Deduplicate first and the cache is far smaller than the corpus.

It is bounded and evicts least-recently-used, because a cache that grows with the corpus defeats the
purpose at a million records.

## Swapping the runtime or tokenizer

`OnnxEmbeddingRuntime` and `HuggingFaceEmbeddingTokenizer` implement `EmbeddingRuntime` and
`EmbeddingTokenizer`. Construct `OnnxEmbeddingVectorizer` directly with your own implementations if
you need a different inference engine, pre-tokenized input, or a tokenizer the DJL binding does not
cover. Pooling, normalisation, caching and fingerprinting stay as they are.

## Runnable example

```bash
./gradlew :examples:run --args="embedding-onnx" \
  -Dskein.onnx.model=e5-small-onnx/model.onnx \
  -Dskein.onnx.tokenizer=e5-small-onnx/tokenizer.json
```

Trains the recipe tagger on your embeddings and scores it against a hand-written held-out set using
wording the rules never mention, next to the hashing baseline. Without the properties it prints the
export command.
