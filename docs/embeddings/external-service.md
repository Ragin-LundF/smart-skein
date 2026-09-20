# Route B — embeddings from an external service

The model runs somewhere else and you call it over HTTP. LM Studio, Ollama, llama.cpp's server,
vLLM, Text Embeddings Inference and the OpenAI API all speak the same request shape, so one client
covers all of them.

This is the fastest way to find out whether embeddings help you. Read
[the identity limitation](#the-limitation-that-decides-production-use) and
[Canary the vectors](#canary-the-vectors) before putting it in production.

## What you get

[`skein-classify-embedding-http`](../../skein-classify-embedding-http), a published adapter over
the JDK's HTTP client. Its only dependency beyond `skein-classify` is a JSON parser, which the
protocol requires.

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-classify-embedding-http")
}
```

The wire format, batching, out-of-order responses, width validation and error handling are covered
by [tests against a real local HTTP server](../../skein-classify-embedding-http/src/test/kotlin/io/skein/classify/embedding/http/infrastructure/HttpEmbeddingVectorizerTest.kt).

The module owns the **protocol**. Retries, backoff, proxies and connection pooling are yours:
implement `EmbeddingTransport`, or pass your own `HttpClient` to `JdkHttpEmbeddingTransport`. There
is no policy here that would suit a local LM Studio and a rate-limited hosted API equally.

## Setting up LM Studio

1. **Install LM Studio** and open the **Discover** tab.
2. **Download an embedding model.** Search for `multilingual-e5-small` — 384 dimensions, around 100
   languages, roughly half a gigabyte. See [Choosing a model](choosing-a-model.md) for alternatives.
   Embedding models are listed separately from chat models; make sure you have an embedding one.
3. **Open the Developer tab** (the server view), load the model, and **Start Server**. The default
   is `http://localhost:1234`.
4. **Note the model identifier** LM Studio shows for the loaded model. That exact string goes into
   `EmbeddingServiceConfig.model`, and it is not guessable — the same weights are
   `multilingual-e5-small-mlx` on a machine that pulled the MLX conversion and something else
   elsewhere. `curl http://localhost:1234/v1/models` lists the ids your server knows. A wrong id
   comes back as HTTP 400 with the server's own message.

   > `/v1/models` lists what is **downloaded**, not what is **loaded**. Unless just-in-time
   > loading is on, an id can be listed and still answer `"No models loaded"` until you run
   > `lms load <id>`. `lms ps` shows what is actually resident.
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

Set `baseUrl = "https://api.openai.com/v1"`, `model = "text-embedding-3-small"` and
`headers = mapOf("Authorization" to "Bearer $token")`. Be deliberate about this one: every record
you classify leaves your infrastructure.

Header values are kept out of the fingerprint, so rotating a token does not invalidate a trained
model, and they are redacted from `EmbeddingServiceConfig.toString()` so a bearer token does not
reach your logs.

## Configure it

```kotlin
import io.skein.classify.embedding.http.domain.EmbeddingProbes
import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.infrastructure.HttpEmbeddingVectorizer

val config = EmbeddingServiceConfig(
    baseUrl = "http://localhost:1234/v1",
    model = "text-embedding-multilingual-e5-small",
    modelRevision = "lmstudio/multilingual-e5-small@1",
    inputPrefix = "passage: ",
    dimension = 384,
    batchSize = 32,
    canaryProbes = EmbeddingProbes.DEFAULT,
)

val vectorizer = HttpEmbeddingVectorizer(config = config)
check(vectorizer.isReachable()) { "no embedding service at ${config.embeddingsUrl()}" }
```

Five fields deserve attention:

- **`modelRevision`** is a label *you* maintain for the exact weights behind `model`. It is a
  declaration, not a verification — see below. Bump it whenever you change or update the served
  model.
- **`canaryProbes`** is the part that actually verifies. Leave it empty and nothing changes from
  earlier releases; set it and a model changed on the server stops the next load. See
  [Canary the vectors](#canary-the-vectors).
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

1. **Use [route A](onnx-local.md) in production** where the choice is open. The model file's bytes
   are hashed, so a swap is impossible to miss rather than merely detectable.
2. **Canary the vectors.** Set `canaryProbes` and the sequence above fails at step 4 instead of
   succeeding. This is what makes route B usable in production; it is covered below.
3. **Pin the service.** Run a fixed model version in a container you control, and treat changing it
   as a deployment that includes bumping `modelRevision` and retraining. Good practice, but it is a
   process, and processes are forgotten.

The base URL and the request headers are deliberately **not** part of the fingerprint: the same
model served from another host, or reached with a rotated token, must keep its identity, or moving
a service between machines would invalidate every trained model for no reason.

## Canary the vectors

Fixed probe texts, embedded when the model is saved and re-embedded when it is loaded. If they have
moved, the load throws `VectorizerCanaryException` instead of scoring.

```kotlin
val config = EmbeddingServiceConfig(
    // ... as above ...
    canaryProbes = EmbeddingProbes.DEFAULT,   // or your own
    canaryTolerance = 0.01,                   // relative L2; the default
)
```

That is the whole change. `saveMultiLabel` captures the references, and `loadMultiLabel` checks
them. A model saved without probes behaves exactly as it always did.

### What it catches, and what it does not

| Change on the server | Caught? |
|---|---|
| A different model loaded under the same name | Yes |
| The same model requantised, `fp16` → `q8_0` | Yes — and it should be; for a classifier trained on its vectors that is a different model |
| The service switched to L2-normalising its output | Yes. This is why drift is relative L2 and not cosine distance: normalising rescales every vector without rotating it, so cosine would see nothing while your linear classifier breaks |
| A fine-tune of the same base model | Yes |
| Different backend, CPU → Metal → CUDA | Usually not. Legitimate numerical noise; widen `canaryTolerance` to about `0.05` if you deliberately move between backends |
| Batch-size and threading differences | No. Accumulation order moves a vector by about `1e-5`, a thousandth of the default tolerance |

The probes are embedded **one per request**, never batched, so batch composition cannot contribute
drift of its own.

### Three things to know before enabling it

- **Loading a model now performs network I/O.** One round trip per probe, and it can fail because
  the service is unreachable — a timeout propagates as itself, not as
  `VectorizerCanaryException`, because "the service is down" and "your model changed" call for
  different responses. `loadMultiLabel(path, vectorizer, verifyCanary = false)` opts out.
- **Probe texts are stored in the model file in clear text.** They are the one thing in a `.skein`
  file that is not an irreversible hash. Use short synthetic sentences written for the purpose,
  never records from your corpus. `EmbeddingProbes.DEFAULT` is three such sentences, in three
  languages, because a probe set confined to one language cannot see a revision that shifts
  another.
- **`saveMultiLabel` re-checks before writing.** A canary taken before training proves nothing
  about a model swapped *during* a long run — half the corpus would be embedded by one model and
  half by another, and a stale reference would still match at load. Saving therefore re-embeds the
  probes and refuses to write if they have already moved.

If the canary fires and the change was deliberate, retrain and save a fresh one. Do not widen the
tolerance: that is turning off the smoke alarm.

Recorded in [ADR 0002](../adr/0002-vector-canary.md).

## Other operational notes

| Concern | What to do |
|---|---|
| Service down mid-run | `vectorizeAll` throws. Embed to a file first for long runs, then train from it. |
| Rate limits | Lower `batchSize`, and add backoff in your own `EmbeddingTransport`. |
| Timeouts | `timeoutSeconds` defaults to 120. Large batches on a loaded CPU can exceed a shorter one. |
| Data residency | Every record's text leaves your process. For a hosted API it leaves your infrastructure. If that is a problem, route A is the answer. |
| Cost | A hosted API charges per token. Deduplicate first — see [Scale](../classify/scale.md). |

## Runnable examples

Two, answering different questions.

```bash
./gradlew :examples:run --args="embedding-service"
```

**Is this worth it?** With a server up it discovers a model, trains the recipe tagger on its
embeddings, and scores it against a hand-written held-out set next to the hashing baseline.
Without one it prints the setup steps.

```bash
./gradlew :examples:run --args="localai"
```

**How do I run it safely?** Discovers the model, calibrates the canary tolerance against it,
trains, saves and reloads, and then stages a model swap so you can watch the canary catch it.
With no server reachable it starts an in-process stub and runs the whole thing against that —
including the swap, which a real server cannot be asked to perform on demand.

Its pieces are written to be lifted:
[`ModelDiscovery`](../../examples/src/main/kotlin/io/skein/examples/localai/ModelDiscovery.kt) for
finding a model that actually embeds, and
[`CanaryCalibrator`](../../examples/src/main/kotlin/io/skein/examples/localai/CanaryCalibrator.kt)
for measuring drift — at startup to choose a tolerance, and on a timer in a long-running service
to notice a model that changed while you held it open.
