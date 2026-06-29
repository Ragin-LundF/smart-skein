package io.skein.examples.regression

import io.skein.classify.application.ClassificationService
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.InMemoryFeatureStore
import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import io.skein.classify.infrastructure.NaiveBayesClassifier

private const val LR_EPOCHS = 10
private const val LR_SEED = 42L
private const val LABEL_COL = 12

private val SCHEMA = Schema.define {
    text(name = "purpose")
    label(name = "category")
}

private val HASHING_CONFIG = HashingConfig(key0 = 0x1A2B3C4DL, key1 = 0x5E6F7A8BL)

private val TRAINING_RECORDS = listOf(
    Record(values = mapOf("purpose" to "monthly rent apartment", "category" to "rent")),
    Record(values = mapOf("purpose" to "rent payment landlord", "category" to "rent")),
    Record(values = mapOf("purpose" to "health insurance premium annual", "category" to "insurance")),
    Record(values = mapOf("purpose" to "insurance premium car policy", "category" to "insurance")),
    Record(values = mapOf("purpose" to "salary october employer payout", "category" to "salary")),
)

private fun printPredictionBlock(header: String, prediction: Prediction) {
    println(header)
    println("Prediction: ${prediction.label.value}  (confidence ${"%.4f".format(prediction.confidence)})")
    prediction.alternatives.forEach { scored ->
        println("  ${scored.label.value.padEnd(LABEL_COL)} ${"%6.4f".format(scored.probability)}")
    }
    println()
}

fun runLogisticRegressionExample() {
    println("=== Logistic Regression vs Naive Bayes ===")
    println()

    val probe = Record(values = mapOf("purpose" to "salary november payment"))

    val nbService = ClassificationService(
        schema = SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HASHING_CONFIG,
        classifier = NaiveBayesClassifier(),
    )
    nbService.learnAll(records = TRAINING_RECORDS)
    val nbPrediction = nbService.classify(record = probe)
    printPredictionBlock(
        header = "--- Naive Bayes ---\nInput     : \"salary november payment\"",
        prediction = nbPrediction,
    )

    val lrStore = InMemoryFeatureStore()
    val lrService = ClassificationService(
        schema = SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HASHING_CONFIG,
        classifier = LogisticRegressionSgdClassifier(
            initialLearningRate = 0.1,
            decayRate = 0.01,
            l2Regularization = 0.001,
        ),
        featureStore = lrStore,
    )
    lrService.learnAll(records = TRAINING_RECORDS)
    lrService.retrain(epochs = LR_EPOCHS, seed = LR_SEED)
    val lrPrediction = lrService.classify(record = probe)
    printPredictionBlock(
        header = "--- Logistic Regression SGD ($LR_EPOCHS epochs, seed=$LR_SEED) ---\n" +
            "Input     : \"salary november payment\"",
        prediction = lrPrediction,
    )

    val delta = lrPrediction.confidence - nbPrediction.confidence
    println("--- Confidence delta ---")
    println("NB  confidence : ${"%.4f".format(nbPrediction.confidence)}")
    println("LR  confidence : ${"%.4f".format(lrPrediction.confidence)}")
    println("Delta (LR - NB): ${if (delta >= 0) "+" else ""}${"%.4f".format(delta)}")
    println()

    println("When to switch:")
    println("  Naive Bayes   — fast, works well on small corpora; start here.")
    println("  Logistic Reg. — better with correlated features; use retrain() when you have enough data.")
    println()

    val margin = { pred: Prediction ->
        if (pred.alternatives.size >= 2) pred.alternatives[0].probability - pred.alternatives[1].probability else 1.0
    }
    println("Top-2 margin  NB=${"%.4f".format(margin(nbPrediction))}  LR=${"%.4f".format(margin(lrPrediction))}")
    println()

    val nbMetrics = nbService.metrics()
    println("Training observations  NB=${nbMetrics.totalObservations}  LR=${lrService.metrics().totalObservations}")
    nbMetrics.perLabelCounts.forEach { (label, count) -> println("  ${label.value}: $count") }
}
