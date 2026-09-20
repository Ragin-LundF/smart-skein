# Learnable token tagging


When layouts vary too much for fixed rules, **train a tagger**. `CrfSequenceLabeler` is a
linear-chain Conditional Random Field: it assigns a `Tag` to every token, learning from labeled
example sequences. Tags are **discovered from your training data** — there's no fixed tag set.

### Train it

```kotlin
val tokenizer = TypedTokenizer()
val labeler = CrfSequenceLabeler()                 // defaults below

// Labeled examples: each token gets a tag. tokens.size must equal tags.size.
val training = listOf(
    tokenizer.tokenize("customer ab12")  to listOf(Tag("KEY"), Tag("VALUE")),
    tokenizer.tokenize("amount 67,89")   to listOf(Tag("KEY"), Tag("VALUE")),
    // ... more examples covering the variation you expect
)

// Online SGD: replay the corpus for several epochs until it converges (~200 is typical for small sets).
repeat(200) { training.forEach { (tokens, tags) -> labeler.learn(tokens, tags) } }

// Predict tags for new token sequences:
val tags = labeler.label(tokenizer.tokenize("customer ab12"))   // [KEY, VALUE]
```

### How the CRF works (data scientists)

- **State features** per token: token type, lowercased word, 3-char prefix & suffix, and the
  neighboring tokens' types (`prevType` / `nextType`, with a `^` boundary marker at the ends).
- **Transition features**: a start score per tag and a `from-tag → to-tag` score for every adjacent
  pair — this is what lets the model learn "a VALUE usually follows a KEY".
- **Decoding**: **Viterbi** finds the single highest-scoring tag sequence.
- **Training**: online SGD on the conditional log-likelihood. The gradient is
  `gold counts − expected counts`, where expected counts come from a numerically stable, log-space
  **forward-backward** pass. New tags are registered the first time they appear in `learn`.

### Hyperparameters

```kotlin
CrfSequenceLabeler(
    initialLearningRate = 0.1,    // base SGD step;  lr(t) = lr0 / (1 + decayRate * step)
    decayRate = 0.0,              // 0.0 = constant LR; >0 anneals over a long run
    l2Regularization = 0.0,       // >0 to regularize weights and curb overfitting
)
```

| Param | Default | Tune toward |
|-------|---------|-------------|
| `initialLearningRate` | `0.1` | ↓ if training is unstable, ↑ if convergence is slow |
| `decayRate` | `0.0` | `>0` for long training runs to settle the weights |
| `l2Regularization` | `0.0` | `>0` on small/noisy data to fight overfitting |

There is **no built-in convergence test** — you control the number of epochs. Small, clean datasets
(a handful of patterns) typically converge within ~200 passes; complex/ambiguous data needs more
examples and more epochs. Hold out some sequences to check tagging accuracy.

### Error behavior

- `learn` throws `IllegalArgumentException` if `tokens.size != tags.size`; an empty token list is a
  no-op.
- `label` throws `IllegalStateException` if called before any training; an empty token list returns
  an empty list.

`SequenceLabeler` is an SPI — swap in your own tagger implementation if you prefer a different model.

---

[Extraction](README.md) · [Slot filling](slots.md) · [Patterns](patterns.md) · [Clustering](clustering.md) · [CRF tagging](crf.md) · [Persistence](persistence.md) · [All documentation](../README.md)
