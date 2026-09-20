# Scale

Training on more than a million records. Every item here is cheaper designed in than retrofitted.

## Do these four things, in this order

Each step makes the next cheaper.

### 1. Mask high-cardinality tokens

```kotlin
import io.skein.classify.application.TokenMasker

val masker = TokenMasker()
val masked = corpus.map { row -> row.copy(featureText = masker.mask(text = row.featureText)) }
```

The single biggest win and the cheapest. A real export is full of tokens occurring exactly once —
reference numbers, mandate ids, card numbers, timestamps. Each becomes its own feature, each is
noise, and together they are most of the feature space.

**Measured: 59% fewer features, 60% smaller model**, for one tokenisation pass.

Structure is preserved rather than discarded: `4711-2024` becomes `<NUMERIC>-<NUMERIC>`, so a
record's shape still distinguishes it from one with no reference at all. Built on `skein-text`'s
`TypedTokenizer`, so date and amount recognition stays consistent with the rest of the library —
supply a tokenizer configured for your locale.

| Token type | Masked | Kept |
|---|---|---|
| `DATE`, `AMOUNT` | always | — |
| `NUMERIC` | runs ≥ 4 digits | shorter quantities |
| `ALPHANUMERIC` | runs ≥ 5 characters | shorter codes |
| `WORD`, `SYMBOL`, `WORD_SYMBOL` | — | always |

### 2. Deduplicate

```kotlin
import io.skein.classify.application.CorpusDeduplicator

val result = CorpusDeduplicator().deduplicate(corpus = corpus)
result.rows            // masked survivors
result.occurrences     // how many originals each stood in for
result.ratio()         // report this on your data
```

Once references are masked away, an enormous share of a real corpus is literally the same record.
**Measured: 400,000 documents collapsed to 21,420** — nineteen to one, with the vocabulary
essentially unchanged.

That particular ratio is inflated by the probe drawing from only 7,319 base documents. Expect a
smaller but still large factor, and **measure it** rather than assuming.

Rows are keyed on masked text **together with the label set**, never text alone. Two rows with
identical wording and different labels are a genuine disagreement in your data — a mislabelling, or
a distinction your features cannot see — and silently discarding one would hide it.

Dropping duplicates also removes a bias towards high-volume sources. If frequency should influence
the fit, keep `occurrences` and weight by it instead.

### 3. Choose the vectorizer by measurement

Feature hashing has one property that matters at this scale: **fitting memory is constant in corpus
size**. There is no vocabulary, so there is nothing that grows.

An explicit vocabulary must hold every term it has seen before it can apply a document-frequency
floor. At a million documents × ~250 n-grams that means holding the unique subset of 250 million
strings before throwing most away — which is a design defect, not a tuning problem, because the cost
is incurred *before* the floor is reached.

Hashing bounds the model; it does not minimise it. Pick the width from measured data:

| Width | Model size, 237 labels, 8% kept |
|---|---|
| `2^18` (262,144) | 37 MB |
| `2^20` (1,048,576) | 151 MB |
| `2^22` (4,194,304) | 606 MB |

### 4. Scale the document-frequency floor

Only relevant if you use [IDF](featurisation.md#idf-over-hashed-buckets). A floor of 2 is right at
seven thousand documents and absurd at a million, where a bucket in two documents is
indistinguishable from a hash collision.

```kotlin
DocumentFrequencyTable.floorFor(documentCount = corpus.size)   // max(2, 0.00002 × N)
```

## What is handled for you

### The design matrix is blocked by rows

A single flat CSR pair caps at `Int.MAX_VALUE` non-zeros. At a measured ~250 active features per
document that ceiling arrives at roughly **8.5 million documents**, and the failure is an opaque
`NegativeArraySizeException` from an overflowed size computation.

`SparseMatrix` splits rows across blocks, so that ceiling does not exist for any corpus that fits in
memory. Values are stored as `float` rather than `double`, halving the largest structure in the
process — the accumulators stay `double`, so accuracy is unaffected.

The one limit blocking cannot remove is a single row wider than a whole block, which fails with a
message naming the block limit rather than an allocation error.

### Training parallelism is bounded by memory

Every concurrent L-BFGS fit holds its own curvature history. The memory is per worker, so
parallelism multiplies it:

| Features | Per worker (history 5) | × 18 workers |
|---|---|---|
| 102,205 | 12 MB | 225 MB |
| 369,529 | 45 MB | 812 MB |
| 4,194,304 | 512 MB | 9.0 GB |

`TrainingParallelism` sizes the pool from available heap and feature count, not core count.
Defaulting to `parallelStream()`'s common pool would make heap exhaustion a function of the
machine's core count — the one variable with nothing to do with whether the memory is there — and it
would bite hardest at the widest feature space, which is exactly the configuration you reach for
when accuracy matters.

Lower `historySize` to 5 or below if you are tight; it costs a little convergence speed.

### Embeddings

If you use an [embedding vectorizer](../embeddings/README.md), featurisation becomes the dominant
cost: at 2 ms per record, a million rows is 33 minutes per training run. Deduplicate first, batch
with `vectorizeAll`, and cache. After masking and deduplication the cache is usually far smaller
than the corpus.

## Known scaling shape

Measured with an explicit-vocabulary pipeline, which is what the hashing recommendation above exists
to avoid:

| Documents | Features | Vocab fit | Matrix build | Heap |
|---|---|---|---|---|
| 7,319 | 118,370 | 0.4 s | 0.2 s | 49 MB |
| 25,000 | 297,236 | 1.4 s | 0.9 s | 157 MB |
| 100,000 | 369,524 | 5.5 s | 3.1 s | 448 MB |
| 400,000 | 369,529 | 20.7 s | 12.3 s | 1,412 MB |
| 1,000,000 | — | **did not complete** | — | **11.6 GB and climbing** |

Clean linear scaling to 400,000, then a wall — caused entirely by vocabulary fitting. With hashing
that step does not exist.

## A checklist before calling it done

- [ ] One million records train within a **declared** heap budget, and the budget is written down.
- [ ] Peak memory is measured and **linear** from 100k to 1M — no superlinear step.
- [ ] Model size is independent of corpus size. Prove it: train on 100k and 1M and compare.
- [ ] `keepFraction` chosen from a measured size/accuracy sweep on your data, not inherited.
- [ ] Deduplication ratio reported on real data, not assumed.
- [ ] A 10M-document run either works or fails with a clear error naming the limit.
- [ ] Inference throughput within ~2× of your baseline.
