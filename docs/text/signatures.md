# Pattern signatures


The ordered sequence of token types — a **content-free structural fingerprint**. Safe to log,
compare across records, and cluster on (`skein-extract`'s `TemplateClusterer` does exactly this).

```kotlin
val tokens = tokenizer.tokenize("payment 1234.56 on 2024-12-31")
val signature = PatternSignature.of(tokens)
signature.render()        // "<word> <amount> <word> <date>"
signature.types           // [WORD, AMOUNT, WORD, DATE]

// Two records with the same layout share a signature even if every value differs:
PatternSignature.of(tokenizer.tokenize("invoice 99,00 on 2025-01-01")).render()
// "<word> <amount> <word> <date>"  ← same fingerprint
```

It's a `@JvmInline value class` over `List<TokenTypeEnum>`, so equality/hashing work out of the box —
you can use signatures as map keys with no boxing cost.

---

[Text foundation](README.md) · [Normalization](normalization.md) · [Tokenization](tokenization.md) · [Signatures](signatures.md) · [Word repair](word-repair.md) · [All documentation](../README.md)
