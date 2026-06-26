# skein-text

Shared text foundation for Skein. Pure, zero heavy dependencies. Depended on by
`skein-classify` and `skein-extract`.

## What it does

- **Normalization** (`TextNormalizer` / `DefaultTextNormalizer`) — control-char removal,
  whitespace collapse, trim, lowercasing. Idempotent.
- **Typed tokenization** (`TypedTokenizer`) — splits text into `Token`s, each tagged with a
  `TokenTypeEnum` (`WORD`, `NUMERIC`, `ALPHANUMERIC`, `DATE`, `AMOUNT`, `SYMBOL`, `WORD_SYMBOL`).
- **Pattern signatures** (`PatternSignature`) — the ordered token-type sequence, a
  content-free structural fingerprint such as `<word> <date> <numeric>`.
- **Broken-word repair** (`BrokenWordRepairer`) — merges wrongly split fragments
  (`"apart ment" → "apartment"`) by re-segmenting with dynamic programming over a self-learned
  `FrequencyModel`; tolerant of typos via bounded edit distance.
- **Frequency model** (`FrequencyModel`) — self-learned word frequencies with a privacy threshold:
  rare words (likely personal data) never count as vocabulary.

## Quick start

```kotlin
val normalizer = DefaultTextNormalizer()
val tokenizer = TypedTokenizer()

val tokens = tokenizer.tokenize(normalizer.normalize("payment 1234.56 on 2024-12-31"))
val signature = PatternSignature.of(tokens).render()   // "<word> <amount> <word> <date>"

val frequencyModel = FrequencyModel().apply { learnAll(listOf("apartment", "apartment")) }
val repaired = BrokenWordRepairer(frequencyModel).repair("apart ment")   // "apartment"
```

## Package layout

```
io.skein.text
├─ domain/          Token, TokenTypeEnum, PatternSignature, FrequencyModel
├─ spi/             TextNormalizer (port)
├─ application/     TypedTokenizer, BrokenWordRepairer
└─ infrastructure/  DefaultTextNormalizer
```

> Token classification is heuristic and tuned for European (notably German) financial text — dates
> with `.`/`-`/`/` separators and comma-decimal amounts; the regex patterns in `TypedTokenizer` are
> the calibration knobs.
