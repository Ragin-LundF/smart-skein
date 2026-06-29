package io.skein.examples.cli

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.ClassifierKindEnum
import io.skein.classify.application.LoadedModel
import io.skein.classify.application.ModelStore
import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.FieldSpec
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.TextField
import io.skein.classify.infrastructure.InMemoryFeatureStore
import java.nio.file.Files

private const val PREDICT_DIVIDER_WIDTH = 65
private const val EXPORT_PREVIEW_LINES = 15

private val SCHEMA = Schema.define {
    text(name = "purpose")
    label(name = "category")
}

private val HASHING_CONFIG = HashingConfig(key0 = 0xFEED_C0DEL, key1 = 0xBAD_BEAFL)

private fun trainingRecords(): List<Record> = listOf(
    Record(values = mapOf("purpose" to "monthly rent apartment", "category" to "rent")),
    Record(values = mapOf("purpose" to "rent payment landlord", "category" to "rent")),
    Record(values = mapOf("purpose" to "health insurance premium", "category" to "insurance")),
    Record(values = mapOf("purpose" to "car insurance policy annual", "category" to "insurance")),
    Record(values = mapOf("purpose" to "salary october employer payout", "category" to "salary")),
    Record(values = mapOf("purpose" to "salary november bonus payment", "category" to "salary")),
)

private fun sampleData(): List<Record> = listOf(
    Record(values = mapOf("purpose" to "rent apartment monthly")),
    Record(values = mapOf("purpose" to "insurance premium car")),
    Record(values = mapOf("purpose" to "salary payout employer")),
    Record(values = mapOf("purpose" to "payment transfer fee")),
)

fun runCliToolDemoExample() {
    println("=== CLI Predict & Export (library APIs) ===")
    println()

    val store = InMemoryFeatureStore()
    val service = ClassificationService(
        schema = SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HASHING_CONFIG,
        featureStore = store,
    )
    service.learnAll(records = trainingRecords())

    val tempPath = Files.createTempFile("model-demo", ".skein")
    try {
        ModelStore.save(
            path = tempPath,
            schema = SCHEMA,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = HASHING_CONFIG,
            observations = store.all(),
        )
        println("Model saved: $tempPath  (${store.size()} observations)")
        println()

        val loaded = ModelStore.load(path = tempPath)
        printPredictionTable(model = loaded)

        println("--- export (CLI: skein export --model model.skein --out model.txt) ---")
        println()
        val exportLines = buildExportLines(model = loaded)
        exportLines.take(EXPORT_PREVIEW_LINES).forEach { line -> println(line) }
        if (exportLines.size > EXPORT_PREVIEW_LINES) {
            println("  ... (${exportLines.size - EXPORT_PREVIEW_LINES} more lines)")
        }
        println()
    } finally {
        Files.deleteIfExists(tempPath)
    }
}

private fun printPredictionTable(model: LoadedModel) {
    println("--- predict (CLI: skein predict --data transactions.csv --model model.skein --out out.csv) ---")
    val predictService = rebuildService(model = model)
    println("%-40s  %-12s  %s".format("purpose", "label", "confidence"))
    println("-".repeat(PREDICT_DIVIDER_WIDTH))
    sampleData().forEach { record ->
        val prediction = predictService.classify(record = record)
        println(
            "%-40s  %-12s  ${"%.4f".format(prediction.confidence)}".format(
                "\"${record["purpose"]}\"",
                prediction.label.value,
            ),
        )
    }
    println()
}

private fun rebuildService(model: LoadedModel): ClassificationService {
    val rebuilt = InMemoryFeatureStore()
    val service = ClassificationService(
        schema = model.schema,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = model.hashingConfig,
        featureStore = rebuilt,
    )
    model.observations.forEach { obs ->
        val labelFieldName = model.schema.labelField.name
        val purposeField = model.schema.fields.firstOrNull { it !is LabelField }?.name ?: "purpose"
        service.learn(
            record = Record(
                values = mapOf(
                    purposeField to obs.label.value,
                    labelFieldName to obs.label.value,
                ),
            ),
        )
    }
    return service
}

private fun buildExportLines(model: LoadedModel): List<String> {
    val lines = ArrayList<String>()
    lines.add("skein-model 1")
    lines.add("classifier ${model.classifier.name}")
    val c = model.hashingConfig
    lines.add(
        "hashing ${c.key0} ${c.key1} ${c.numFeatures} " +
            "${c.charNgramMin} ${c.charNgramMax} ${c.wordNgramMin} ${c.wordNgramMax}",
    )
    model.schema.fields.forEach { field ->
        lines.add("field ${fieldTypeName(field)} ${field.name} ${field.sensitivity.name}")
    }
    lines.add("---")
    model.observations.forEach { obs ->
        val features = obs.features
        val tokens = buildString {
            for (i in features.indices.indices) {
                if (i > 0) append(' ')
                append(features.indices[i]).append(':').append(features.values[i])
            }
        }
        lines.add("${obs.label.value}\t$tokens")
    }
    return lines
}

private fun fieldTypeName(field: FieldSpec): String {
    return when (field) {
        is TextField -> "TEXT"
        is CategoricalField -> "CATEGORICAL"
        is NumericField -> "NUMERIC"
        is IdentifierField -> "IDENTIFIER"
        is LabelField -> "LABEL"
    }
}
