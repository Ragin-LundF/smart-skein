# examples

Runnable Skein samples. **Not published.** Depends on all library modules (`skein-text`,
`skein-classify`, `skein-extract`).

> **Audience:** anyone who learns best from working code. This is the recommended starting point —
> it shows how the three building blocks compose into a real pipeline.

## Run it

```bash
./gradlew :examples:run          # main class: io.skein.examples.MainKt
./gradlew :examples:test         # asserts the end-to-end behavior
```

Expected output:

```
purpose : AIG-Life 67.89 Geico-Auto 120.00 insurance premium
  category   : insurance (confidence 1.00)
  extracted  : insurer = AIG-Life
  extracted  : amount = 67.89
  extracted  : insurer = Geico-Auto
  extracted  : amount = 120.00
purpose : rent apartment CustomerNumber CD456 monthly
  category   : rent (confidence 1.00)
  extracted  : customer = CD456
purpose : salary november payout
  category   : salary (confidence 0.xx)
  extracted  : (no fields routed for this category)
```

---

## Tutorial: `TransactionCategorizationExample`

The example implements **classify → route → extract** over bank-transaction purpose text. Walk it
top to bottom — each step maps to one of the library modules.

### Step 0 — define the schema

```kotlin
private val schema = Schema.define {
    text(name = "purpose")          // the free-text we classify on → becomes feature text
    identifier(name = "iban")       // PII by default → excluded from features
    label(name = "category")        // the target label
}
```

`purpose` carries the signal; `iban` is present but, being an `identifier` (PII), never enters the
feature text; `category` is what the model predicts. (See `skein-classify` for the field types.)

### Step 1 — build the engine and train it

```kotlin
private val engine = ClassificationService(
    schema = schema,
    privacyMode = PrivacyModeEnum.FEATURES_ONLY,
    hashingConfig = HashingConfig.randomKey(),     // fresh key per run — fine for a demo
)
private val extractor = SlotExtractor()

init { train() }                                   // train as soon as the example is constructed
```

`train()` feeds eight labeled examples across three categories:

```kotlin
engine.learnAll(listOf(
    sample("AIG-Life 67.89 Geico-Auto 120.00 insurance premium", "insurance"),
    sample("Allstate-Home 90.00 insurance premium",              "insurance"),
    sample("insurance premium annual policy",                    "insurance"),
    sample("rent apartment CustomerNumber AB123 monthly",        "rent"),
    sample("monthly rent apartment CustomerNumber XY999",        "rent"),
    sample("rent apartment payment",                             "rent"),
    sample("salary october payout",                              "salary"),
    sample("monthly salary payment employer",                    "salary"),
))
// sample(purpose, category) builds Record(mapOf("purpose" to ..., "iban" to "DE00", "category" to ...))
```

That's the whole training step — Naive Bayes learns in a single pass, so there's no `retrain` loop
here. (Switch to `LogisticRegressionSgdClassifier` + `retrain(epochs)` if you experiment with more
data; see `skein-classify`.)

> ⚠️ **`HashingConfig.randomKey()` is for demos only.** The feature indices differ every run, so a
> model trained here cannot be persisted and reloaded. In production pass a **fixed secret key**
> (`HashingConfig(key0 = …, key1 = …)`) so the model is reproducible.

### Step 2 — classify, route, extract

```kotlin
fun process(purpose: String): PipelineResult {
    // CLASSIFY: predict the category. The dummy "iban" satisfies the schema but is excluded from features.
    val prediction = engine.classify(Record(mapOf("purpose" to purpose, "iban" to "DE00")))

    // ROUTE + EXTRACT: pick category-specific slot rules, then pull the structured values.
    val extraction = extractor.extract(text = purpose, slots = slotsFor(prediction.label))

    return PipelineResult(prediction.label, prediction.confidence, extraction)
}
```

The **routing** is a plain `when` on the predicted label — each category gets its own extraction
rules:

```kotlin
private fun slotsFor(label: Label): List<SlotDefinition> = when (label.value) {
    "insurance" -> listOf(                          // recurring (insurer, amount) pairs
        RepeatingGroupSlot("policies", listOf(
            GroupComponent("insurer", TokenTypeEnum.WORD_SYMBOL),
            GroupComponent("amount",  TokenTypeEnum.AMOUNT),
        )),
    )
    "rent" -> listOf(                               // the value after the "CustomerNumber" keyword
        KeyAnchoredSlot("customer", anchor = "CustomerNumber", targetType = TokenTypeEnum.ALPHANUMERIC),
    )
    else -> emptyList()                             // "salary" → nothing to extract
}
```

This is the key idea: **the classifier decides _what kind_ of text this is, which decides _which_
extraction rules to run.** Insurance text yields insurer/amount pairs; rent text yields a customer
number; salary text needs no extraction. (See `skein-extract` for the slot kinds.)

### What comes back — `PipelineResult`

```kotlin
data class PipelineResult(
    val category: Label,            // predicted label
    val confidence: Double,         // 0.0..1.0
    val extracted: ExtractionResult // structured fields for that category
)
```

The end-to-end behavior is asserted in `TransactionCategorizationExampleTest` — read it alongside
this for the exact expected fields.

---

## Adapting it to your data

1. **Redefine the schema** for your fields; mark anything sensitive as `identifier` / `PII`.
2. **Replace the training samples** with your labeled records (or load them via
   `RecordImportService` and `learnAll`). More examples per category → better accuracy; consider
   `ActiveLearningSelector` to choose what to label next.
3. **Write `slotsFor` cases** per category using the slot kinds from `skein-extract`, or train a
   `CrfSequenceLabeler` when fixed rules aren't enough.
4. **Use a fixed `HashingConfig` key** and a durable `FeatureStore` (`skein-store-postgres`) once you
   want the model to persist.
</content>
