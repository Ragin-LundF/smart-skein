# Slot filling


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

[Extraction](README.md) · [Slot filling](slots.md) · [Patterns](patterns.md) · [Clustering](clustering.md) · [CRF tagging](crf.md) · [Persistence](persistence.md) · [All documentation](../README.md)
