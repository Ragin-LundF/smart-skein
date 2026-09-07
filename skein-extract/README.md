# skein-extract

Pull **structured values** out of unstructured text via typed-token patterns and slot filling.
Unlike `skein-classify`, this module returns the **real values** — it does not hash or destroy them —
and **persists nothing unless you explicitly save a trained tagger** (see
[Persisting a trained tagger](#persisting-a-trained-tagger--crfmodelstore)). Depends on `skein-text`.

> **Audience:** developers defining extraction rules, and data scientists who want to discover layout
> templates automatically or **train** the learnable CRF token tagger.

## Two ways to extract

| Approach | Tool | Needs training? | Use when |
|----------|------|-----------------|----------|
| **Rule-based slots** | `SlotExtractor` + slot definitions | no | the layout is known/stable (keys, positions, repeating groups) |
| **Learned tagging** | `CrfSequenceLabeler` | yes (label examples) | the layout varies and you can provide labeled token sequences |
| **Unsupervised discovery** | `TemplateClusterer` | no | you don't yet know the layouts and want to find them |

All three operate on the typed tokens from `skein-text`, so the `TokenTypeEnum` vocabulary
(`WORD`, `DATE`, `AMOUNT`, `WORD_SYMBOL`, …) is the shared language. Skim `skein-text`'s README first
if those types are unfamiliar.

## Installation

```kotlin
dependencies {
    implementation("io.skein:skein-extract:<version>")     // align via skein-bom
}
```

---

## 1. Slot filling — `SlotExtractor` (rule-based, no training)

Define *what* you want and *where* it sits relative to the token stream; the extractor returns the
matching values with their source spans. Three slot kinds cover most layouts.

```kotlin
val extractor = SlotExtractor()                       // uses TypedTokenizer internally

val result = extractor.extract(
    "CustomerNumber: AB12345 AIG-Life 67.89 Geico-Auto 120.00",
    listOf(
        KeyAnchoredSlot("customer", anchor = "CustomerNumber", targetType = TokenTypeEnum.ALPHANUMERIC),
        RepeatingGroupSlot("policies", listOf(
            GroupComponent("insurer", TokenTypeEnum.WORD_SYMBOL),
            GroupComponent("amount", TokenTypeEnum.AMOUNT),
        )),
    ),
)

result.first("customer")?.value     // "AB12345"
result.valuesOf("amount")           // ["67.89", "120.00"]
result.valuesOf("insurer")          // ["AIG-Life", "Geico-Auto"]
```

### Slot kinds — `SlotDefinition`

| Slot | Constructor | Extracts | Notes |
|------|-------------|----------|-------|
| `PositionalSlot` | `(name, tokenIndex)` | the token at a fixed index | nothing if the index is out of bounds |
| `KeyAnchoredSlot` | `(name, anchor, targetType = null)` | the value following a keyword | tolerant of punctuation glued to the anchor (`"CustomerNumber:"` still matches anchor `"CustomerNumber"`) |
| `RepeatingGroupSlot` | `(name, components)` | recurring runs of components | each occurrence emits one field per component, linked by `groupIndex`; needs ≥1 component |

**`KeyAnchoredSlot.targetType`:**
- `null` → take the token immediately after the anchor.
- a `TokenTypeEnum` → take the **first following token of that type** (skip over noise in between).

**`RepeatingGroupSlot`** scans left-to-right: when the next tokens match all components in order it
emits one group and advances past it; otherwise it advances one token and retries. Each emitted
field carries a `groupIndex` (0, 1, 2, …) so you can re-pair `insurer[0]`↔`amount[0]`.

### What you get back — `ExtractionResult` / `ExtractedField`

```kotlin
data class ExtractedField(
    val name: String,
    val value: String,          // the REAL extracted text, unmodified
    val span: SourceSpan,       // half-open [startOffset, endOffset) in the source string
    val confidence: Double = 1.0,
    val groupIndex: Int? = null, // occurrence index for repeating groups; null otherwise
)
```

`ExtractionResult` helpers:

```kotlin
result.fields                  // List<ExtractedField>, in source order
result.first("customer")       // first field with that name, or null
result.valuesOf("amount")      // all values for that name, in order
```

Rule-based matches are deterministic, so `confidence` is `1.0`. (The CRF labeler is where
sub-`1.0` confidences come from.)

> **Tip:** `extract` is overloaded — pass either a raw `String` (it tokenizes for you) or a
> pre-tokenized `List<Token>` if you already ran `skein-text`'s tokenizer with a specific mode.

---

## 2. Low-level matching — `TokenPattern` + `PatternMatcher`

When you need "regex over token *types*" rather than named slots, build a `TokenPattern` and find its
spans. This is the matching primitive slots are built on; use it directly to locate or validate
structure.

```kotlin
val pattern = TokenPattern.of {
    type(TokenTypeEnum.WORD)            // exactly one
    optional(TokenTypeEnum.SYMBOL)      // zero or one
    oneOrMore(TokenTypeEnum.AMOUNT)     // one or more (greedy)
    zeroOrMore(TokenTypeEnum.WORD)      // zero or more (greedy)
}

val matcher = PatternMatcher()
matcher.findAll("sum 1.00 2.00 3.00", pattern)   // [0..3]  — token-index IntRanges, leftmost & non-overlapping
matcher.matchesFully(tokens, pattern)            // true only if the pattern spans ALL tokens
```

### DSL quantifiers — `QuantifierEnum`

| Builder call | Quantifier | Meaning |
|--------------|-----------|---------|
| `type(t)` | `ONE` | exactly one token of type `t` |
| `optional(t)` | `OPTIONAL` | zero or one |
| `zeroOrMore(t)` | `ZERO_OR_MORE` | zero or more (greedy, backtracks) |
| `oneOrMore(t)` | `ONE_OR_MORE` | one or more (greedy, backtracks) |

Matching is **greedy with backtracking**: a `oneOrMore`/`zeroOrMore` element consumes as much as it
can, then gives tokens back if the rest of the pattern would otherwise fail. `findAll` returns
inclusive token-index ranges; non-matching input yields an empty list (never an error).

---

## 3. Discover layouts — `TemplateClusterer` (unsupervised)

Don't know your layouts yet? Cluster texts by their `PatternSignature` (the content-free token-type
fingerprint from `skein-text`). Texts with identical structure land together.

```kotlin
val clusterer = TemplateClusterer()
val clusters = clusterer.cluster(listOf(
    "booked 2024-12-31 12.50",      // <word> <date> <amount>
    "reversal 2025-01-01 99.00",    // <word> <date> <amount>
    "hello world",                  // <word> <word>
))

clusters.first().signature.render()   // "<word> <date> <amount>"  (largest cluster first)
clusters.first().members              // ["booked 2024-12-31 12.50", "reversal 2025-01-01 99.00"]
```

Returns `List<TemplateCluster>` (`signature` + member `members`), **sorted largest-first**. Use it to
find the dominant layouts in a corpus, then write a `SlotExtractor` rule per cluster — or to feed the
CRF below with examples drawn from each layout.

---

## 4. Learnable token tagging — `CrfSequenceLabeler` (trainable)

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

## 5. Persisting a trained tagger — `CrfModelStore`

A CRF that only lives in memory loses its training on exit. `CrfModelStore` writes one to a single
file and reads it back, hyperparameters and progress included, so training resumes rather than
restarting.

```kotlin
val labeler = CrfSequenceLabeler()
repeat(200) { training.forEach { (tokens, tags) -> labeler.learn(tokens = tokens, tags = tags) } }

CrfModelStore.save(path = Path("tagger.skeincrf"), snapshot = labeler.snapshot())

val restored = CrfSequenceLabeler.from(snapshot = CrfModelStore.load(path = Path("tagger.skeincrf")))
restored.label(tokens = tokens)                    // same output as the original
restored.learn(tokens = more, tags = moreTags)     // and training continues from the saved step
```

`CrfModelStore.metadata(path)` reads the header alone — retention policy, writer version, creation
time — without inflating the weights.

### ⚠️ A saved CRF contains your training text

This is the one place where Skein's "never stored in clear text" property does **not** hold. A CRF
learns features keyed by the token text itself (`word=`, plus three-character `prefix=`/`suffix=`
affixes), so the weight table is a lexicon of the training corpus. Anyone with the file can
decompress it and read that vocabulary. `skein-classify` is different — its features are
irreversible keyed hashes.

| Retention | What is written | Privacy | Cost |
|---|---|---|---|
| `ALL_FEATURES` | every weight | **none** — every token seen even once is on disk | the only mode that restores a bit-identical model |
| `FREQUENT_ONLY` *(default)* | lexical features seen at least `minLexicalOccurrences` times (default 2), plus all structure | **reduced, not eliminated** — a name occurring twice survives | negligible; a weight from one occurrence is statistically worthless anyway |
| `STRUCTURAL_ONLY` | token-type structure and transitions only | **zero clear text** | loses generalization from word shape, e.g. recognizing an unseen `-ing` word by its suffix |

The default is the safer mode deliberately: `ALL_FEATURES` as a default would turn a
privacy-preserving library into a PII exporter the first time anyone called `save`.

### File format `SKCR`

```
offset 0   4 bytes  magic 'S' 'K' 'C' 'R'
offset 4   1 byte   format major (breaking)
offset 5   1 byte   format minor (additive)
offset 6   ...      GZIP( payload )
```

Hand-rolled over `DataOutputStream`, so this module keeps its single dependency on `skein-text`.
The reader accepts any minor version — including a newer one — and never asserts end-of-stream, so
fields a later writer appends are ignored rather than fatal. A differing major is refused with a
message naming both versions. Enum values are written as stable codes, never ordinals, so
reordering an enum constant cannot silently reinterpret an old file.

### Error behavior

- `save` throws `IllegalArgumentException` for an untrained labeler or a `minLexicalOccurrences`
  below 1.
- `load` throws `IllegalArgumentException` for a foreign magic, an unreadable major version, or a
  truncated or corrupt payload.
- `CrfSequenceLabeler.from` validates the snapshot — non-empty and duplicate-free tag order, a
  non-negative step, and no weight referencing an unknown tag.

---

## Package layout

```
io.skein.extract
├─ domain/          TokenPattern + TokenPatternBuilder DSL, PatternElement, QuantifierEnum,
│                   SlotDefinition (PositionalSlot, KeyAnchoredSlot, RepeatingGroupSlot, GroupComponent),
│                   ExtractedField, SourceSpan, ExtractionResult, TemplateCluster, Tag
├─ application/     PatternMatcher, SlotExtractor, TemplateClusterer
├─ spi/             SequenceLabeler (port)
└─ infrastructure/  CrfSequenceLabeler, CrfModelStore, CrfModelSnapshot,
                    CrfModelMetadata, FeatureRetentionEnum
```

| Goal | Reach for |
|------|-----------|
| Pull a known value by position / keyword / repeating group | `SlotExtractor` + slot definitions |
| Match raw token-type structure | `TokenPattern` + `PatternMatcher` |
| Find layouts you don't know yet | `TemplateClusterer` |
| Tag tokens when rules can't keep up | `CrfSequenceLabeler` (train it) |
| Keep a trained tagger across restarts | `CrfModelStore` (mind the retention modes) |

See [`examples`](../examples) for `SlotExtractor` driven by a classifier's prediction — the full
**classify → route → extract** pipeline.

