# Choosing an embedding model

For classification you want the **smallest model whose vectors still separate your labels**. Unlike
retrieval, you are not ranking millions of candidates by fine-grained similarity — you are handing
features to a linear model that then learns from your labels. A compact encoder usually loses very
little, and it is the difference between 300 and 5,000 records per second.

## What actually matters

| Property | Why it matters here |
|---|---|
| **Languages** | A model that never saw your language produces vectors that separate nothing. This is the first filter, not the last. |
| **Dimensions** | Sets the classifier's size (`dims × labels × 4 bytes`) and its scoring cost. 384 is plenty for most classification; 1024 is rarely worth it. |
| **Parameters / file size** | Drives latency and memory. Under ~150 MB is comfortable on a CPU. |
| **Max sequence length** | Text beyond it is truncated. 512 tokens covers most records; check if yours are long. |
| **Required prefix** | E5 needs `passage: `. Omitting it costs accuracy silently. |
| **Licence** | Apache-2.0 and MIT are safe; some strong models are non-commercial. |

Dimensions deserve emphasis because the instinct is backwards. More dimensions is not better for
classification — it is more parameters for your linear model to fit from the same number of labelled
examples. With a few hundred examples per label, 384 dimensions is usually the better choice even
when a 1024-dimension variant scores higher on retrieval leaderboards.

## Concrete starting points

Sizes are approximate for fp32 and roughly quarter that when quantised to int8. **Verify the
details on the model card before committing** — families get revised, and licences change.

### Multilingual, small — start here

| Model | Dims | Approx. size | Notes |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | ~470 MB | ~100 languages. Strong quality per byte. **Requires `passage: ` / `query: ` prefixes.** MIT. |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~470 MB | ~50 languages, no prefix needed, very widely used. Apache-2.0. |
| `intfloat/multilingual-e5-base` | 768 | ~1.1 GB | The step up when `small` is not separating your labels. Same prefix rule. |

**The default recommendation is `multilingual-e5-small`**: broad language coverage, 384 dimensions,
and small enough to run comfortably on a CPU. Its only trap is the prefix, and that is one line of
configuration.

### English-only

| Model | Dims | Approx. size | Notes |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90 MB | The classic small baseline. Fast, no prefix. Apache-2.0. |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~550 MB | Supports truncating dimensions (Matryoshka), so you can trade width for speed after training. |

### When latency dominates

Static embedding models (the Model2Vec family, e.g. `minishlab/potion-base-8M` and its multilingual
siblings) are a token-embedding lookup plus a mean — no transformer forward pass. That puts them at
roughly 100–200 µs per record, an order of magnitude faster than MiniLM, with a real but modest
quality drop.

They are worth a serious look when you are anywhere near a throughput ceiling. The tiny model used
in this repository's own ONNX integration tests is exactly this architecture, which is why those
tests exercise the real code path rather than a mock.

## How to decide, concretely

1. **Filter by language.** Non-negotiable.
2. **Take the smallest candidate** — 384 dimensions, under ~500 MB.
3. **Measure it on your data**, against the hashing baseline:
   ```bash
   ./gradlew :examples:run --args="embedding-service"
   ```
   The comparison in [`EmbeddingComparison`](../../examples/src/main/kotlin/io/skein/examples/embedding/EmbeddingComparison.kt)
   trains both ways and scores both on held-out wording the training data does not contain.
4. **Only if it is not separating your labels, go bigger.** Check macro-F1, not micro — a bigger
   model usually helps the rare labels first, and micro-F1 hides that.
5. **Pin the model.** Record the exact repository, revision and quantisation in
   `modelRevision` (route B) or ship the file itself (route A).

## A note on leaderboards

MTEB and similar rankings measure retrieval and semantic similarity. You are doing neither: you are
producing features for a supervised linear model that gets to learn from your labels. A model three
places lower on a leaderboard and three times faster is very often the better choice here. Treat
rankings as a shortlist, and your own comparison as the decision.
