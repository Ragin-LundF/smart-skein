# skein-extract

Pull structured values out of text via typed-token patterns and slot filling. Returns the **real
values** (it does not hash or destroy them) and **persists nothing by default**. Depends on `skein-text`.

## What it does

- **`TokenPattern`** + DSL — "regex over token types" with quantifiers (`type`, `optional`,
  `zeroOrMore`, `oneOrMore`). `PatternMatcher` finds leftmost, non-overlapping matches via greedy
  backtracking.
- **Slot filling** (`SlotExtractor`):
  - `PositionalSlot` — the token at a fixed index.
  - `KeyAnchoredSlot` — the value following a keyword (`"CustomerNumber" → following ALPHANUMERIC`),
    tolerant of trailing punctuation.
  - `RepeatingGroupSlot` — recurring component runs (each `WORD_SYMBOL + AMOUNT → (insurer, amount)`),
    with parts linked by `groupIndex`.
- **`TemplateClusterer`** — unsupervised grouping of texts by `PatternSignature` (recurring layouts).
- **`ExtractionResult`** — extracted fields with their values, source spans and confidence.

## Quick start

```kotlin
val extractor = SlotExtractor()
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
```

## Package layout

```
io.skein.extract
├─ domain/          TokenPattern + DSL, SlotDefinition hierarchy, ExtractedField, ExtractionResult, TemplateCluster …
└─ application/     PatternMatcher, SlotExtractor, TemplateClusterer
```

## Learnable token tagging (Phase 8)

- **`SequenceLabeler`** SPI port — `learn(tokens, tags)` / `label(tokens)`.
- **`CrfSequenceLabeler`** — a linear-chain CRF: state features (token type + lowercased word) and
  transition features (prev-tag → tag, plus a start transition), Viterbi decoding, and online-SGD
  training on the conditional log-likelihood (gradient = gold counts − expected counts from a
  log-space forward-backward pass). Tags (`Tag`) are discovered from the training data.

```kotlin
val labeler = CrfSequenceLabeler()
repeat(200) { trainingSequences.forEach { (tokens, tags) -> labeler.learn(tokens, tags) } }
val tags = labeler.label(tokenizer.tokenize("customer ab12"))   // [KEY, VALUE]
```
