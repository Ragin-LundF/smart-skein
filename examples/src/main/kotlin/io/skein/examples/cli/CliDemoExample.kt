package io.skein.examples.cli

import io.skein.classify.application.ActiveLearningSelector
import io.skein.classify.application.ClassificationService
import io.skein.classify.application.SchemaInference
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.UncertaintyStrategyEnum

private const val FIELD_TYPE_COL = 12

private fun labeledData(): List<Record> = listOf(
    Record(values = mapOf("purpose" to "monthly rent apartment", "amount" to "850.00", "category" to "rent")),
    Record(values = mapOf("purpose" to "rent payment landlord", "amount" to "850.00", "category" to "rent")),
    Record(values = mapOf("purpose" to "health insurance premium", "amount" to "120.50", "category" to "insurance")),
    Record(values = mapOf("purpose" to "car insurance annual", "amount" to "540.00", "category" to "insurance")),
    Record(values = mapOf("purpose" to "salary october payout", "amount" to "3200.00", "category" to "salary")),
    Record(values = mapOf("purpose" to "salary november employer", "amount" to "3200.00", "category" to "salary")),
)

private fun unlabeledData(): List<Record> = listOf(
    Record(values = mapOf("purpose" to "payment transfer fee", "amount" to "10.00")),
    Record(values = mapOf("purpose" to "monthly transfer", "amount" to "500.00")),
    Record(values = mapOf("purpose" to "annual payment premium", "amount" to "200.00")),
    Record(values = mapOf("purpose" to "bonus payout november", "amount" to "1000.00")),
)

fun runCliDemoExample() {
    println("=== CLI Equivalent Demo (library APIs only) ===")
    println()

    val labeled = labeledData()
    val schema = SchemaInference().infer(records = labeled, labelField = "category")
    println("Inferred schema:")
    schema.fields.forEach { field ->
        val typeName = field.javaClass.simpleName.removeSuffix("Field").lowercase()
        println("  ${typeName.padEnd(FIELD_TYPE_COL)} ${field.name}")
    }
    println()
    println("CLI equivalent: skein infer --data transactions.csv --label category --out schema.json")
    println()

    val service = ClassificationService(
        schema = schema,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HashingConfig(key0 = 0xABCD1234L, key1 = 0xEF567890L),
    )
    service.learnAll(records = labeled)

    val metrics = service.metrics()
    println("Trained model: ${metrics.totalObservations} observations")
    metrics.perLabelCounts.forEach { (label, count) -> println("  ${label.value}: $count") }
    println()
    println("CLI equivalent: skein train --data transactions.csv --schema schema.json --model model.skein")
    println()

    val selector = ActiveLearningSelector(service = service)
    val toReview = selector.selectForReview(
        candidates = unlabeledData(),
        limit = 2,
        strategy = UncertaintyStrategyEnum.MARGIN,
    )
    println("--- Active learning: top-2 uncertain records ---")
    toReview.forEach { candidate ->
        println("  purpose : \"${candidate.record["purpose"]}\"")
        println("  guess   : ${candidate.prediction.label.value} (${"%.4f".format(candidate.prediction.confidence)})")
        println("  margin  : ${"%.4f".format(candidate.margin)}")
        println()
    }
    println("CLI equivalent: skein label --data unlabeled.csv --model model.skein --limit 2 --strategy MARGIN")
    println()

    println("--- Predictions for all unlabeled candidates ---")
    unlabeledData().forEach { record ->
        val prediction = service.classify(record = record)
        println("  \"${record["purpose"]}\"  ->  ${prediction.label.value} (${"%.4f".format(prediction.confidence)})")
    }
    println()
    println("CLI equivalent: skein predict --data unlabeled.csv --model model.skein --out predictions.csv")
    println()
}
