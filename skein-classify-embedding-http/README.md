# skein-classify-embedding-http

An optional adapter implementing `skein-classify`'s `Vectorizer` port against any OpenAI-compatible
`/embeddings` endpoint, so a classifier can generalise past literal wording without exporting a
model or loading a native runtime.

Contains no learning logic. Depends on `skein-classify` because that is where the port lives.

## Use it when

Hashed n-grams match text as written. A model trained on `eggplant` has learned nothing about
`aubergine` — the two share no meaningful n-gram. Reach for an embedding when your records use
varied wording for the same thing, when labels came from a rule engine and the model must
generalise *past* the rules, or when examples per label are too few for literal overlap to carry.

Reach for **this** adapter rather than
[`skein-classify-embedding-onnx`](../skein-classify-embedding-onnx) when you want to be running in
ten minutes, when you are still choosing a model, or when a service already exists in your
infrastructure. Works unchanged against LM Studio, Ollama, llama.cpp's server, vLLM, Text
Embeddings Inference and the OpenAI API.

**Route A is still the recommendation for production.** See below.

## What it costs

| | |
|---|---|
| Per record | One network round trip. Batch, or it is unusable |
| Text | **Leaves your process.** For a hosted API it leaves your infrastructure |
| Availability | The service is now in your training path, and in your load path if you use a canary |

`EmbeddingServiceConfig.batchSize` defaults to 32 inputs per request. `vectorizeAll` is the method
that matters; `vectorize` sends one text and exists only because the port requires it.

This adapter implements `BatchVectorizer`, so library code that featurises a whole corpus — cross-
validation in particular — batches automatically without knowing the concrete type.

## At a glance

```kotlin
import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.infrastructure.HttpEmbeddingVectorizer

val vectorizer = HttpEmbeddingVectorizer(
    config = EmbeddingServiceConfig.lmStudioMultilingualE5Small(),
)

val features = vectorizer.vectorizeAll(texts = corpusTexts)
// train exactly as with hashed features — nothing downstream changes
```

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-classify-embedding-http")
}
```

The only dependency beyond `skein-classify` is a JSON parser, which the embeddings protocol
requires. There is no native binary here and nothing to export.

## Model identity, and the canary that checks it

A model trained on embeddings from model X and scored with embeddings from model Y does not fail.
It returns confident, wrong labels, with nothing in the output to notice.

`skein-classify-embedding-onnx` rules that out by hashing the model file's bytes. **A service
cannot.** It returns vectors, and nothing in the protocol says which weights produced them. So this
adapter offers two things, and they are not equally strong:

**`modelRevision` is a declaration.** A label you maintain for the exact weights behind `model`. It
enters the fingerprint, so changing it invalidates models trained against the old value — but
nothing forces you to change it, and a stale one is how a silently-swapped model reaches
production.

**`canaryProbes` is a verification.** Fixed texts embedded when the model is saved and re-embedded
when it is loaded. A model changed on the server moves them, and `ModelStore.loadMultiLabel` throws
`VectorizerCanaryException` instead of scoring.

```kotlin
EmbeddingServiceConfig(
    baseUrl = "http://localhost:1234/v1",
    model = "text-embedding-multilingual-e5-small",
    modelRevision = "lmstudio/multilingual-e5-small@1",
    inputPrefix = "passage: ",
    dimension = 384,
    canaryProbes = EmbeddingProbes.DEFAULT,
)
```

Two consequences worth knowing before you enable it:

- **Loading a model now performs network I/O** — one round trip per probe — and can fail because
  the service is down. That is the trade: a loud failure at load instead of a silent one at
  inference. `loadMultiLabel(path, vectorizer, verifyCanary = false)` opts out.
- **Probe texts are stored in the model file in clear text.** They are the one thing in a `.skein`
  file that is not an irreversible hash. Use short synthetic sentences you wrote for the purpose,
  never records from your corpus.

The base URL and the request headers are deliberately **outside** the fingerprint: moving a service
between hosts, or rotating a token, must not invalidate every model you have trained.

## Bring your own transport

`JdkHttpEmbeddingTransport` is a working default, not the only option. It has no retry policy, no
backoff and no circuit breaker, because nothing suits a local LM Studio and a rate-limited hosted
API equally. This module owns the protocol; implement `EmbeddingTransport` to wrap whichever client
your service already uses, or pass your own `HttpClient` for a proxy, an executor or a TLS context.
Auth headers go in `EmbeddingServiceConfig.headers`, whose values are redacted from `toString`.

## Documentation

| | |
|---|---|
| [Overview](../docs/embeddings/README.md) | When embeddings are worth it, the two routes, what can fail silently |
| [Route B — external service](../docs/embeddings/external-service.md) | LM Studio, Ollama, OpenAI-compatible servers, the canary |
| [Route A — local ONNX](../docs/embeddings/onnx-local.md) | Export, integrate, deploy, cache |
| [Choosing a model](../docs/embeddings/choosing-a-model.md) | Small multilingual options and how to judge them |

Related: [Featurisation](../docs/classify/featurisation.md) ·
[Persistence](../docs/classify/persistence.md) · [All documentation](../docs/README.md)

## Runnable example

```bash
./gradlew :examples:run --args="embedding-service"
```

With a server up it trains the recipe tagger on service embeddings and scores it against the hashing
baseline. Without one it prints the setup steps.
