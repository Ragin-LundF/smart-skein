# examples

Runnable Skein demonstrations. **Not published.** Depends on `skein-text`, `skein-classify`,
and `skein-extract`.

Each use case is a self-contained example of how to wire up a classifier for a specific domain.
Run any of them from the project root via Gradle:

```bash
./gradlew :examples:run --args="<use-case> [args...]"
```

---

## Use cases

### `transaction` — classify → route → extract

Classifies bank-transaction purpose text into categories (insurance / rent / salary), then routes
each category to specific extraction rules that pull out the structured values that matter for it.

Shows the full **classify → route → extract** pipeline using `ClassificationService`, `NaiveBayesClassifier`,
and `SlotExtractor` in a single end-to-end flow. Training data is inline; no external files needed.

```bash
./gradlew :examples:run --args="transaction"
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

### `logwatch` — anomaly detection in log files

Trains a classifier from a **keyword rules CSV** (`anomaly_type, level, message`) and then scans
log files line by line, flagging anything that looks like an anomaly.

Shows how to:
- Define a schema with a text field (`message`) and a categorical field (`level`)
- Generate training records from a rules file instead of hand-labeling data
- Save a trained model with `ModelStore.save` and reload it with `ModelStore.load`
- Apply the model at scan time without retraining

```bash
# Step 1: train on the bundled rules and save the model
./gradlew :examples:run --args="logwatch train"

# Step 1 (custom rules): train on your own rules CSV
./gradlew :examples:run --args="logwatch train --rules /path/to/rules.csv"

# Step 2: scan the bundled sample log (prints anomalies to stdout)
./gradlew :examples:run --args="logwatch scan --sample"

# Step 2 (real log): scan a log file and write anomalies to CSV
./gradlew :examples:run --args="logwatch scan --log /var/log/app.log --out anomalies.csv"

# Adjust the minimum confidence threshold (default 0.5)
./gradlew :examples:run --args="logwatch scan --sample --confidence 0.7"
```

**Rules CSV format** (`anomaly_type,level,message`):
each row defines one labeled training example — the `message` column is the keyword text that
characterizes that anomaly type at the given log level.

```csv
anomaly_type,level,message
NORMAL,INFO,started
NORMAL,DEBUG,processing
DB_ERROR,ERROR,SQLException
TIMEOUT,WARN,timeout
SECURITY,ERROR,access denied
OOM,ERROR,OutOfMemoryError
CONNECTION_FAIL,ERROR,Connection refused
```

The bundled `rules.csv` and `sample.log` live under
`src/main/resources/logwatch/` and are used automatically when no `--rules` / `--log` path is given.

---

## Tutorial: `TransactionCategorizationExample`

The transaction example implements **classify → route → extract** over bank-transaction purpose
text. Walk it top to bottom — each step maps to one of the library modules.

### Step 0 — define the schema

```kotlin
private val schema = Schema.define {
    text(name = "purpose")          // the free-text we classify on → becomes feature text
    identifier(name = "iban")       // PII by default → excluded from features
    label(name = "category")        // the target label
}
```

`purpose` carries the signal; `iban` is present but, being an `identifier` (PII), never enters the
feature text; `category` is what the model predicts.

### Step 1 — build the engine and train it

```kotlin
private val engine = ClassificationService(
    schema = schema,
    privacyMode = PrivacyModeEnum.FEATURES_ONLY,
    hashingConfig = HashingConfig.randomKey(),     // fresh key per run — fine for a demo
)

init { train() }

// train() calls engine.learnAll(listOf(sample(...), ...)) with 8 labeled examples
```

> ⚠️ **`HashingConfig.randomKey()` is for demos only.** The feature indices differ every run, so a
> model trained here cannot be persisted and reloaded. In production pass a **fixed secret key**
> (`HashingConfig(key0 = …, key1 = …)`) so the model is reproducible across restarts.

### Step 2 — classify, route, extract

```kotlin
fun process(purpose: String): PipelineResult {
    val prediction = engine.classify(Record(mapOf("purpose" to purpose, "iban" to "DE00")))
    val extraction = extractor.extract(text = purpose, slots = slotsFor(prediction.label))
    return PipelineResult(prediction.label, prediction.confidence, extraction)
}
```

`slotsFor` is a plain `when` on the label: each category gets its own extraction rules.
The classifier decides *what kind* of text this is; that decision picks *which* rules to run.

---

## Tutorial: `LogAnomalyDetector`

The log-watch example shows the **train from rules → save → load → scan** pattern.

### Step 0 — define the schema

```kotlin
val LOG_SCHEMA: Schema = Schema.define {
    text("message")       // the log message text (keyword used as training signal)
    categorical("level")  // ERROR / WARN / INFO / DEBUG — a separate categorical feature
    label("anomaly_type") // what the model predicts
}
```

### Step 1 — generate training records from the rules CSV

```kotlin
fun trainFromRulesCsv(csv: String): Pair<ClassificationService, InMemoryFeatureStore> {
    val store = InMemoryFeatureStore()
    val service = ClassificationService(
        schema = LOG_SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HASHING_CONFIG,         // fixed key so the saved model is reloadable
        classifier = NaiveBayesClassifier(),
        featureStore = store,                   // pass explicitly so we can hand it to ModelStore.save
    )
    val records = parseRulesCsv(csv).map { row -> Record(values = row) }
    service.learnAll(records = records)
    return service to store
}
```

### Step 2 — save and reload

```kotlin
// Save
ModelStore.save(path, LOG_SCHEMA, ClassifierKindEnum.NAIVE_BAYES, HASHING_CONFIG, store.all())

// Load and rebuild
val model = ModelStore.load(path)
val store = InMemoryFeatureStore().also { it.addAll(model.observations) }
val service = ClassificationService(schema = model.schema, ..., featureStore = store)
service.retrain(epochs = 1)   // replays observations through a fresh NaiveBayes
```

### Step 3 — classify log lines

```kotlin
fun ClassificationService.classifyLine(line: String): LogPrediction? {
    val (level, message) = parseLine(line) ?: return null   // null if no log level found
    val result = classify(record = Record(values = mapOf("message" to message, "level" to level)))
    return LogPrediction(line, level, result.label.value, result.confidence)
}
```

---

## Adapting to your data

| What you want | Where to look |
|---|---|
| Change the schema (fields, types, PII exclusions) | `Schema.define { ... }` |
| Load training records from a file instead of inline | `RecordImportService`, `learnAll` |
| Switch to logistic regression for more data | `LogisticRegressionSgdClassifier`, `retrain(epochs)` |
| Save and reload models across restarts | `ModelStore.save`, `ModelStore.load`, fixed `HashingConfig` |
| Active-learning: pick what to label next | `ActiveLearningSelector` |
| Extract structured values after classification | `SlotExtractor`, `skein-extract` |
