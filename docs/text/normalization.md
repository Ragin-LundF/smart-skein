# Normalization


`DefaultTextNormalizer` applies four conservative, **idempotent** steps in this exact order:

| Step | What | Example |
|------|------|---------|
| 1 | Replace control chars (`\p{Cntrl}`) with a space | `"a\tb"` → `"a b"` |
| 2 | Collapse runs of whitespace (`\s+`) to one space | `"a   b"` → `"a b"` |
| 3 | Trim leading/trailing whitespace | `" a b "` → `"a b"` |
| 4 | Lowercase | `"Apart Ment"` → `"apart ment"` |

```kotlin
val normalizer = DefaultTextNormalizer()
normalizer.normalize("  Payment\t 1234.56  ON  2024-12-31 ")   // "payment 1234.56 on 2024-12-31"

// Idempotent: normalize(normalize(x)) == normalize(x)
```

`TextNormalizer` is an SPI (port). Supply your own implementation if you need locale-specific
folding (e.g. accent stripping); the constructors that accept a normalizer
(`HashingVectorizer`, etc.) take the interface, not the concrete class.

> **Note:** normalization is deliberately conservative — it never deletes word characters or
> punctuation. Anything more aggressive (stemming, accent folding) is your call to add via a custom
> `TextNormalizer`.

---

[Text foundation](README.md) · [Normalization](normalization.md) · [Tokenization](tokenization.md) · [Signatures](signatures.md) · [Word repair](word-repair.md) · [All documentation](../README.md)
