package io.skein.examples.activelearning

import io.skein.classify.application.ActiveLearningSelector
import io.skein.classify.application.ClassificationService
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.UncertaintyStrategyEnum

private const val LABEL_COL_WIDTH = 12

private val SCHEMA = Schema.define {
    text(name = "purpose")
    label(name = "category")
}

private val HASHING_CONFIG = HashingConfig(key0 = 0xDEADBEEFL, key1 = 0xCAFEBABEL)

private fun trainingRecords(): List<Record> = listOf(
    Record(values = mapOf("purpose" to "monthly rent apartment", "category" to "rent")),
    Record(values = mapOf("purpose" to "insurance premium annual health", "category" to "insurance")),
    Record(values = mapOf("purpose" to "salary october employer payout", "category" to "salary")),
)

private fun candidateRecords(): List<Record> = listOf(
    Record(values = mapOf("purpose" to "rent payment landlord")),
    Record(values = mapOf("purpose" to "car insurance policy renewal")),
    Record(values = mapOf("purpose" to "salary november bonus")),
    Record(values = mapOf("purpose" to "payment transfer")),
    Record(values = mapOf("purpose" to "transfer fee")),
    Record(values = mapOf("purpose" to "monthly payment")),
)

private fun printAlternatives(prediction: Prediction) {
    prediction.alternatives.forEach { scored ->
        println("  ${scored.label.value.padEnd(LABEL_COL_WIDTH)} ${"%6.4f".format(scored.probability)}")
    }
}

fun runActiveLearningExample() {
    println("=== Active Learning ===")
    println()

    val service = ClassificationService(
        schema = SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HASHING_CONFIG,
    )
    service.learnAll(records = trainingRecords())

    val metricsAfterTraining = service.metrics()
    println("Initial training: ${metricsAfterTraining.totalObservations} observations")
    metricsAfterTraining.perLabelCounts.forEach { (label, count) -> println("  ${label.value}: $count") }
    println()

    val selector = ActiveLearningSelector(service = service)
    val candidates = candidateRecords()

    println("--- Strategy: MARGIN (top-2 uncertain) ---")
    val marginCandidates = selector.selectForReview(
        candidates = candidates,
        limit = 2,
        strategy = UncertaintyStrategyEnum.MARGIN,
    )
    marginCandidates.forEach { candidate ->
        println("  purpose : \"${candidate.record["purpose"]}\"")
        println("  guess   : ${candidate.prediction.label.value} (${"%.4f".format(candidate.prediction.confidence)})")
        println("  margin  : ${"%.4f".format(candidate.margin)}")
        println()
    }

    val mostUncertain = marginCandidates.first()
    println("Most uncertain record:")
    println("  \"${mostUncertain.record["purpose"]}\"")
    printAlternatives(prediction = mostUncertain.prediction)
    println()

    println("--- Feedback: correct label = rent ---")
    service.feedback(record = mostUncertain.record, correctLabel = Label(value = "rent"))
    val metricsAfterFeedback = service.metrics()
    println("After feedback: ${metricsAfterFeedback.totalObservations} observations")
    metricsAfterFeedback.perLabelCounts.forEach { (label, count) -> println("  ${label.value}: $count") }
    println()

    println("--- Metrics diff ---")
    val rentBefore = metricsAfterTraining.perLabelCounts[Label("rent")]
    val rentAfter = metricsAfterFeedback.perLabelCounts[Label("rent")]
    println("Before: total=${metricsAfterTraining.totalObservations}  rent=$rentBefore")
    println("After : total=${metricsAfterFeedback.totalObservations}  rent=$rentAfter")
    println()

    println("--- All 3 uncertainty strategies (limit=1 each) ---")
    UncertaintyStrategyEnum.entries.forEach { strategy ->
        val picked = selector.selectForReview(candidates = candidates, limit = 1, strategy = strategy)
        val top = picked.firstOrNull()
        println("  $strategy -> \"${top?.record?.get("purpose")}\"  margin=${top?.margin?.let { "%6.4f".format(it) }}")
    }
    println()
}
