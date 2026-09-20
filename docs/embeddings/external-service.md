# Route B — embeddings from an external service

The model runs somewhere else and you call it over HTTP. LM Studio, Ollama, llama.cpp's server,
vLLM, Text Embeddings Inference and the OpenAI API all speak the same request shape, so one client
covers all of them.

This is the fastest way to find out whether embeddings help you. Read
[the identity limitation](#the-limitation-that-decides-production-use) before putting it in
production.

## What you get

`HttpEmbeddingVectorizer` lives in [`examples`](../../examples/src/main/kotlin/io/skein/examples/embedding),
not in a published module, because it is 120 lines over the JDK's HTTP client with no dependency
worth shipping. Copy it into your codebase and adjust it — authentication headers, retries, a
proxy — rather than consuming it as a library.

It is covered by [tests against a real local HTTP server](../../examples/src/test/kotlin/io/skein/examples/embedding/HttpEmbeddingVectorizerTest.kt),
so the wire format, batching, out-of-order responses and error handling are verified rather than
assumed.

## Setting up LM Studio

1. **Install LM Studio** and open the **Discover** tab.
2. **Download an embedding model.** Search for `multilingual-e5-small` — 384 dimensions, around 100
   languages, roughly half a gigabyte. See [Choosing a model](choosing-a-model.md) for alternatives.
   Embedding models are listed separately from chat models; make sure you have an embedding one.
3. **Open the Developer tab** (the server view), load the model, and **Start Server**. The default
   is `http://localhost:1234`.
4. **Note the model identifier** LM Studio shows for the loaded model. That exact string goes into
   `EmbeddingServiceConfig.model`.
5. **Check it answers:**

   ```bash
   curl http://localhost:1234/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model":"text-embedding-multilingual-e5-small","input":["passage: hello"]}'
   ```

   A JSON body with a `data` array containing an `embedding` means you are ready. Count the floats
   in it — that is your `dimension`.

### Ollama instead

```bash
ollama pull snowflake-arctic-embed2
ollama serve
```

Then set `baseUrl = "http://localhost:11434/v1"` and `model = "snowflake-arctic-embed2"`. Everything
else is identical.

### The OpenAI API instead

Set `baseUrl = "https://api.openai.com/v1"` and `model = "text-embedding-3-small"`, and add an
`Authorization: Bearer …` header to the request builder in `HttpEmbeddingVectorizer`. Be deliberate
about this one: every record you classify leaves your infrastructure.

## Configure it

```kotlin
import io.skein.examples.embedding.EmbeddingServiceConfig
import io.skein.examples.embedding.HttpEmbeddingVectorizer

val config = EmbeddingServiceConfig(
    baseUrl = "http://localhost:1234/v1",
    model = "text-embedding-multilingual-e5-small",
    modelRevision = "lmstudio/multilingual-e5-small@1",
    inputPrefix = "passage: ",
    dimension = 384,
    batchSize = 32,
)

val vectorizer = HttpEmbeddingVectorizer(config = config)
check(vectorizer.isReachable()) { "no embedding service at ${config.embeddingsUrl()}" }
```

Four fields deserve attention:

- **`modelRevision`** is a label *you* maintain for the exact weights behind `model`. It is the only
  record of which weights produced a vector — see below. Bump it whenever you change or update the
  served model.
- **`inputPrefix`** must match the model family. E5 needs `passage: ` for classification. Most
  others need `""`.
- **`dimension`** can be omitted and probed from the first response, but setting it turns a model
  swapped for one of a different width into an immediate failure rather than a model trained on the
  wrong feature space.
- **`batchSize`** trades request size against round trips. 32 is a reasonable default for a local
  server.

`isReachable()` at startup turns "the service is down" into a clear failure at boot instead of a
stack trace forty minutes into a training run.

## Train

```kotlin
val texts = corpus.map { row -> row.featureText }
val features = vectorizer.vectorizeAll(texts = texts)   // batched, not one call per record

val model = LbfgsMultiLabelLearner(
    featureCount = vectorizer.dimension(),
    keepFraction = 1.0,
).fit(
    observations = corpus.indices.map { index ->
        MultiLabeledFeatures(features = features[index], labels = corpus[index].labels)
    },
)
```

Use `vectorizeAll`. `vectorize` sends one record per round trip and exists only because the
`Vectorizer` port requires it.

**Embed once, reuse.** Every training run, every fold, every hyperparameter candidate re-embeds the
same corpus otherwise. For a corpus you will train on repeatedly, embed once and hold the vectors:

```kotlin
val cachedFeatures: List<FeatureVector> = vectorizer.vectorizeAll(texts = texts)
// reuse cachedFeatures across folds and hyperparameter candidates
```

At 5 ms per round trip, 100,000 records is around eight minutes — acceptable once, painful per
candidate.

## Score

```kotlin
ModelStore.saveMultiLabel(
    path = Path.of("tickets.skein"),
    schema = schema,
    model = model as MultiLabelLogisticClassifier,
    vectorizer = vectorizer,
)

// later, at inference
val loaded = ModelStore.loadMultiLabel(path = Path.of("tickets.skein"), vectorizer = vectorizer)
val prediction = loaded.classifier.predict(
    features = vectorizer.vectorize(text = record.featureText),
    threshold = 0.5,
)
```

The service must be reachable at inference time, and its latency is now inside your request budget.
Batch your inference path too if you serve more than a record at a time.

## The limitation that decides production use

The fingerprint covers the model name, your declared `modelRevision`, the input prefix and the
width — everything that changes the vector **and that the client can see**. Nothing in the
embeddings protocol reveals which weights answered.

So this sequence passes every check and produces wrong labels:

1. You train a model against `multilingual-e5-small` served by LM Studio.
2. Someone updates the model in LM Studio.
3. `modelRevision` still says `@1`, so the fingerprint still matches.
4. `loadMultiLabel` accepts the model. Inference returns confident, wrong labels. Nothing logs.

Three ways to live with it, in order of strength:

1. **Use [route A](onnx-local.md) in production.** The model file's bytes are hashed, so this cannot
   happen. Keep route B for experimentation.
2. **Pin the service.** Run a fixed model version in a container you control, and treat changing it
   as a deployment that includes bumping `modelRevision` and retraining.
3. **Canary the vectors.** Keep a handful of fixed probe texts with their embeddings from training
   time, re-embed them at startup, and fail if they have moved beyond a tolerance. Roughly twenty
   lines, and it converts a silent failure into a loud one.

The base URL is deliberately **not** part of the fingerprint: the same model served from another
host must keep its identity, or moving a service between machines would invalidate every trained
model for no reason.

## Other operational notes

| Concern | What to do |
|---|---|
| Service down mid-run | `vectorizeAll` throws. Embed to a file first for long runs, then train from it. |
| Rate limits | Lower `batchSize` and add backoff in the client you copied. |
| Timeouts | `timeoutSeconds` defaults to 120. Large batches on a loaded CPU can exceed a shorter one. |
| Data residency | Every record's text leaves your process. For a hosted API it leaves your infrastructure. If that is a problem, route A is the answer. |
| Cost | A hosted API charges per token. Deduplicate first — see [Scale](../classify/scale.md). |

## Runnable example

```bash
./gradlew :examples:run --args="embedding-service"
```

With a server up it trains the recipe tagger on service embeddings and scores it against a
hand-written held-out set, next to the hashing baseline. Without one it prints the setup steps.
