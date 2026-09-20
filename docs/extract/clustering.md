# Discovering layouts


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

[Extraction](README.md) · [Slot filling](slots.md) · [Patterns](patterns.md) · [Clustering](clustering.md) · [CRF tagging](crf.md) · [Persistence](persistence.md) · [All documentation](../README.md)
