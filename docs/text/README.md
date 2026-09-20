# skein-text

The shared text foundation for Skein. **Pure, zero heavy dependencies, no training data shipped.**
Every other module (`skein-classify`, `skein-extract`) builds on the four primitives here:
normalize → tokenize → fingerprint → repair.

> **Audience:** developers wiring text through the pipeline, and data scientists who need to know
> exactly how raw strings become typed tokens and privacy-preserving signatures before any model
> sees them.

## Pages

| | |
|---|---|
| [Normalization](normalization.md) | `TextNormalizer`, `DefaultTextNormalizer`, the four folding steps |
| [Typed tokenization](tokenization.md) | `TypedTokenizer`, token types, `TokenPatternConfig`, locale presets |
| [Pattern signatures](signatures.md) | `PatternSignature` — content-free structural fingerprints |
| [Broken-word repair](word-repair.md) | `BrokenWordRepairer` and the self-learned `FrequencyModel` |

## Mental model

```
raw string ──normalize──▶ clean string ──tokenize──▶ List<Token> ──┬─▶ PatternSignature  (structure only)
                                                                    └─▶ feature text / extraction
                              ▲
                  BrokenWordRepairer + FrequencyModel
                  (optional repair of split words before tokenizing)
```

- **Normalization** strips noise so the same text always looks the same.
- **Tokenization** assigns each token a *structural type* (`WORD`, `DATE`, `AMOUNT`, …) — this is the
  privacy-friendly view: it captures *shape*, not content.
- **Pattern signatures** are the ordered list of those types — a content-free fingerprint you can
  safely log, compare, and cluster.
- **Broken-word repair** is a *self-learned* clean-up step (the only learnable component here): it
  re-joins words that OCR or bad exports split apart.

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-text")
}
```

## End-to-end usage

```kotlin
val normalizer = DefaultTextNormalizer()
val tokenizer = TypedTokenizer(mode = TokenizationModeEnum.WHITESPACE)

val model = FrequencyModel(minKeepFrequency = 3).apply {
    learnAll(corpus.flatMap { it.split(' ') })       // self-learn vocabulary from your corpus
}
val repairer = BrokenWordRepairer(model)

val clean = repairer.repair(normalizer.normalize(rawText))
val tokens = tokenizer.tokenize(clean)
val signature = PatternSignature.of(tokens).render()
```

This `clean → tokens → signature` flow is exactly what `skein-classify` (feature text) and
`skein-extract` (slot matching, clustering) consume.

## Package layout

```
io.skein.text
├─ domain/          Token, TokenTypeEnum, TokenizationModeEnum, TokenPatternConfig, PatternSignature, FrequencyModel
├─ spi/             TextNormalizer (port)
├─ application/     TypedTokenizer, BrokenWordRepairer, SymSpellIndex
└─ infrastructure/  DefaultTextNormalizer
```

| Type | Layer | Key API |
|------|-------|---------|
| `TextNormalizer` | spi | `normalize(String): String` |
| `DefaultTextNormalizer` | infrastructure | the 4-step pipeline above |
| `TypedTokenizer` | application | `tokenize(String): List<Token>`; ctor `mode`, `patterns` |
| `Token` | domain | `text`, `type`, `startOffset`, `endOffset` |
| `TokenPatternConfig` | domain | `typedRules`; presets `GERMAN` (default), `US` |
| `PatternSignature` | domain | `of(tokens)`, `render()`, `types` |
| `FrequencyModel` | domain | `learn`/`learnAll`/`frequency`/`isKnown`/`knownWords`/`serialize`/`deserialize` |
| `BrokenWordRepairer` | application | `repair(String): String`; ctor knobs |

> **Calibration reminder:** token classification is heuristic and locale-tuned. The `TokenPatternConfig`
> passed to `TypedTokenizer` and the privacy threshold in `FrequencyModel` are the knobs that adapt
> Skein to your data — treat them as the first thing to tune when results look off.

---

[Normalization](normalization.md) · [Tokenization](tokenization.md) · [Signatures](signatures.md) · [Word repair](word-repair.md) · [All documentation](../README.md)
