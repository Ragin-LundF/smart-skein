# Schema and records

## Describing your data

A `Schema` declares what a record holds. Field types are not decoration — they decide what becomes a
feature and what never leaves your process.

```kotlin
import io.skein.classify.domain.Schema
import io.skein.classify.domain.SensitivityEnum

val schema = Schema.define {
    text(name = "subject")
    text(name = "body")
    categorical(name = "component")
    numeric(name = "ageInDays")
    identifier(name = "reporterEmail", sensitivity = SensitivityEnum.PII)
    label(name = "tags")
}
```

Exactly one `label` field is required.

### Field types

| Type | Use for | Contributes to features |
|---|---|---|
| `text` | Free text: descriptions, subjects, bodies | yes |
| `categorical` | Values from a small fixed set: a component, a channel | yes |
| `numeric` | Quantities: amounts, durations, counts | yes, as text |
| `identifier` | Account numbers, emails, references | yes **unless** marked `PII` |
| `label` | The ground truth | never |

### Keeping personal data out

```kotlin
identifier(name = "reporterEmail", sensitivity = SensitivityEnum.PII)
```

A `PII` field is excluded from the feature text entirely. It is not hashed, not stored in a model,
and not reachable from an explanation. `RecordMapper` drops it before featurisation, so there is no
path by which it reaches a model file.

Mark it on any field that identifies a person. High-cardinality identifiers are usually worth
excluding anyway — see [Scale](scale.md), where masking them cut the feature count by 59%.

## Records

A record is a map:

```kotlin
import io.skein.classify.domain.Record

val record = Record(
    values = mapOf(
        "subject" to "login fails after update",
        "body" to "since yesterday I cannot sign in on mobile",
        "component" to "auth",
        "reporterEmail" to "someone@example.com",   // present, never featurised
        "tags" to "BUG",                              // omit when predicting
    ),
)
```

For prediction the label field may be absent. For `learn` it must be present.

## Don't have a schema yet?

```kotlin
import io.skein.classify.application.SchemaInference

val proposed = SchemaInference().infer(records = sampleRecords)
```

`SchemaInference` guesses field types from values. Read what it proposes before using it: it cannot
know which of your fields are personal data, so every `PII` marking is yours to add.

## Validating input

```kotlin
import io.skein.classify.application.SchemaValidator

val result = SchemaValidator(schema = schema).validate(record = record)
result.isValid
result.errors      // missing required fields, unparseable numerics
result.warnings    // unknown fields, empty values
```

## Bulk import

For a stream — a CSV, a database cursor — with validation in one pass and constant memory:

```kotlin
import io.skein.classify.application.RecordImportService
import io.skein.classify.spi.RecordSource

val source = object : RecordSource {
    override fun stream(): Sequence<Record> = csvRows.map { row -> Record(values = row) }
}

val result = RecordImportService(schema = schema).importFrom(source = source)

result.accepted        // List<MappedRecord> — validated, PII already stripped
result.rejected        // List<RejectedRecord> — each with its reason
result.warnings
```

`MappedRecord` is `(featureText, label?)`: the mapped, PII-stripped view ready to featurise.
Rejections are returned rather than thrown, so one malformed row does not abort an import of a
million.
