# skein-classify-embedding-onnx

An optional adapter implementing `skein-classify`'s `Vectorizer` port with an external ONNX
sentence-embedding model, so a classifier can generalise past literal wording.

Contains no learning logic. Depends on `skein-classify` because that is where the port lives.

## Use it when

Hashed n-grams match text as written. A model trained on `eggplant` has learned nothing about
`aubergine` — the two share no meaningful n-gram. Reach for an embedding when:

- Your records use varied wording for the same thing: synonyms, dialects, several languages, free
  text from different people.
- Labels came from a rule engine and the model must generalise *past* the rules.
- Examples per label are few and literal overlap is too sparse to learn from.

**Stay with feature hashing** when your text is templated, when throughput is measured in thousands
of records per second, or when feature material must stay irreversible and local.

## What it costs

| Vectorizer | Per record, one core | Throughput |
|---|---|---|
| `HashingVectorizer` | 41 µs | ~24,000/s |
| Static embeddings | ~100–200 µs | 5,000–10,000/s |
| MiniLM-class transformer, int8 | ~1–3 ms | 300–1,000/s per core |

The classifier itself gets *smaller*: 384 dimensions × 237 labels × 4 bytes is 364 KB, against
17 MB for a sparse n-gram model. The artifact that matters becomes the embedding model.

**Measure before adopting.** `./gradlew :examples:run --args="embedding-onnx"` trains the same
tagger both ways and scores both on held-out wording the training data never contains.

## At a glance

```kotlin
import io.skein.classify.embedding.onnx.application.EmbeddingCache
import io.skein.classify.embedding.onnx.infrastructure.OnnxEmbeddingVectorizer
import java.nio.file.Path

OnnxEmbeddingVectorizer.open(
    modelPath = Path.of("e5-small-onnx/model.onnx"),
    tokenizerPath = Path.of("e5-small-onnx/tokenizer.json"),
    cache = EmbeddingCache(),
).use { vectorizer ->
    val features = vectorizer.vectorizeAll(texts = corpusTexts)   // batched
    // train exactly as with hashed features — nothing downstream changes
}
```

Export a model once with Hugging Face Optimum:

```bash
optimum-cli export onnx --model intfloat/multilingual-e5-small --task feature-extraction e5-small-onnx/
```

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-classify-embedding-onnx")
}
```

This module carries the ONNX Runtime and tokenizer native binaries. It is separate from
`skein-classify` precisely so a consumer using feature hashing never inherits them.

## Model identity is the point

A model trained on embeddings from model X and scored with embeddings from model Y does not fail. It
returns confident, wrong labels, with nothing in the output to notice.

This adapter's fingerprint digests **the model file's contents** — not its name, not a version
string — along with the tokenizer, the pooling strategy, the normalisation and the width. A model
swapped in place under an unchanged name changes the digest, and `ModelStore.loadMultiLabel` refuses
it.

An embedding *service* cannot offer that guarantee; see the comparison in the docs.

## Documentation

| | |
|---|---|
| [Overview](../docs/embeddings/README.md) | When embeddings are worth it, the two routes, what can fail silently |
| [Route A — local ONNX](../docs/embeddings/onnx-local.md) | Export, integrate, deploy, cache |
| [Route B — external service](../docs/embeddings/external-service.md) | LM Studio, Ollama, OpenAI-compatible servers |
| [Choosing a model](../docs/embeddings/choosing-a-model.md) | Small multilingual options and how to judge them |

Related: [Featurisation](../docs/classify/featurisation.md) ·
[Architecture](../docs/architecture.md) · [All documentation](../docs/README.md)
