# skein-extract

Pulling structured values *out of* text: typed-token patterns, slot filling, layout clustering, and
a trainable CRF token tagger with its own model format.

Unlike classification, this returns real values rather than hashes. Depends on `skein-text`.

## Use it when

- You need the *value*, not a category: an amount, a date, a reference, a name.
- The text has recognisable structure but no fixed format — several layouts, or layouts you have not
  enumerated.
- You want to discover those layouts from a corpus rather than writing them by hand.
- Patterns are not enough and you need a model that learns from token context.

## At a glance

```kotlin
import io.skein.extract.application.ExtractionService

val result = ExtractionService(schema = extractionSchema).extract(text = documentLine)
result.fields.forEach { field -> println("${field.name} = ${field.value}") }
```

## Installation

```kotlin
dependencies {
    implementation(platform("io.github.ragin-lundf:skein-bom:<version>"))
    implementation("io.github.ragin-lundf:skein-extract")
}
```

## A privacy difference worth knowing

Classification features are irreversible keyed hashes. **The CRF tagger here is different**: it
learns features keyed by the token text itself, so a saved CRF model contains fragments of the
training text unless written with `FeatureRetentionEnum.STRUCTURAL_ONLY`.

If that matters for your data, read the persistence section before saving a tagger.

## Documentation

**[Full documentation →](../docs/extract/README.md)** — the pattern DSL, positional and
key-anchored slots, template clustering, training and persisting the CRF tagger, and the feature
retention modes.

Related: [Architecture](../docs/architecture.md) · [All documentation](../docs/README.md)
