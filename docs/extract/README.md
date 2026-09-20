# skein-extract

Pull **structured values** out of unstructured text via typed-token patterns and slot filling.
Unlike `skein-classify`, this module returns the **real values** — it does not hash or destroy them —
and **persists nothing unless you explicitly save a trained tagger** (see
[Persisting a trained tagger](persistence.md)). Depends on `skein-text`.

> **Audience:** developers defining extraction rules, and data scientists who want to discover layout
> templates automatically or **train** the learnable CRF token tagger.

## Pages

| | |
|---|---|
| [Slot filling](slots.md) | `SlotExtractor` — positional and key-anchored extraction, no training |
| [Patterns](patterns.md) | `TokenPattern` and `PatternMatcher`, the DSL underneath |
| [Clustering](clustering.md) | `TemplateClusterer` — discover layouts in a mixed corpus |
| [CRF tagging](crf.md) | `CrfSequenceLabeler` — a trainable tagger for what rules cannot reach |
| [Persistence](persistence.md) | `CrfModelStore`, the `SKCR` format, and what a saved tagger contains |

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
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-extract")
}
```

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

See [`examples`](../../examples) for `SlotExtractor` driven by a classifier's prediction — the full
**classify → route → extract** pipeline.

---

[Slot filling](slots.md) · [Patterns](patterns.md) · [Clustering](clustering.md) · [CRF tagging](crf.md) · [Persistence](persistence.md) · [All documentation](../README.md)
