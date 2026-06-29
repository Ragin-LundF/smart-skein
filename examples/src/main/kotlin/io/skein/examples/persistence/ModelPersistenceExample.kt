package io.skein.examples.persistence

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.ClassifierKindEnum
import io.skein.classify.application.ModelStore
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.InMemoryFeatureStore
import io.skein.classify.infrastructure.NaiveBayesClassifier
import java.nio.file.Files

// Fixed keys are required for persistence: the same key produces the same feature indices
// when the model is reloaded, so predictions are reproducible across JVM runs.
private val HASHING_CONFIG = HashingConfig(
    key0 = 0x736b65696eL,
    key1 = 0x6d6f64656cL,
)

private val SCHEMA = Schema.define {
    text(name = "purpose")
    identifier(name = "iban")
    label(name = "type")
}

private fun rec(purpose: String, iban: String, type: String): Record =
    Record(values = mapOf("purpose" to purpose, "iban" to iban, "type" to type))

private val TRAINING_RECORDS = listOf(
    rec(purpose = "rent payment monthly apartment", iban = "DE89370400440532013000", type = "rent"),
    rec(purpose = "monthly rent apartment CustomerNumber AB123", iban = "DE44500105175407324931", type = "rent"),
    rec(purpose = "rent payment due", iban = "DE12345678901234567890", type = "rent"),
    rec(purpose = "salary october payout employer", iban = "DE55667788990011223344", type = "salary"),
    rec(purpose = "monthly salary payment", iban = "DE99887766554433221100", type = "salary"),
    rec(purpose = "insurance premium AIG life policy", iban = "DE11223344556677889900", type = "insurance"),
    rec(purpose = "insurance premium Geico auto annual", iban = "DE00112233445566778899", type = "insurance"),
)

fun runModelPersistenceExample() {
    println("=== Model Persistence Example ===")
    println()

    // 1. Train with an explicit InMemoryFeatureStore so we can hand its observations to ModelStore.
    val store = InMemoryFeatureStore()
    val original = ClassificationService(
        schema = SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HASHING_CONFIG,
        classifier = NaiveBayesClassifier(),
        featureStore = store,
    )
    original.learnAll(records = TRAINING_RECORDS)

    println("Trained model — observations stored: ${store.size()}")
    val metricsSummary = original.metrics().perLabelCounts.entries
        .joinToString { (label, count) -> "${label.value}=$count" }
    println("Metrics before save: $metricsSummary")
    println()

    // 2. Save to a temp file.
    val modelPath = Files.createTempFile("model", ".skein")
    try {
        ModelStore.save(
            path = modelPath,
            schema = SCHEMA,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = HASHING_CONFIG,
            observations = store.all(),
        )
        println("Model saved to: $modelPath (${Files.size(modelPath)} bytes)")
        println()

        val loaded = ModelStore.load(path = modelPath)
        val restored = rebuildFromModel(loaded = loaded)
        printRoundTripComparison(original = original, restored = restored)
    } finally {
        Files.deleteIfExists(modelPath)
        println()
        println("Temp file deleted.")
    }
}

private fun rebuildFromModel(loaded: io.skein.classify.application.LoadedModel): ClassificationService {
    val restoredStore = InMemoryFeatureStore()
    restoredStore.addAll(observations = loaded.observations)
    val restored = ClassificationService(
        schema = loaded.schema,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = loaded.hashingConfig,
        classifier = NaiveBayesClassifier(),
        featureStore = restoredStore,
    )
    restored.retrain(epochs = 1)
    println("Loaded model — observations restored: ${restoredStore.size()}")
    println("Classifier kind: ${loaded.classifier}")
    println()
    return restored
}

private fun printRoundTripComparison(original: ClassificationService, restored: ClassificationService) {
    val probes = listOf(
        Record(values = mapOf("purpose" to "rent apartment payment", "iban" to "DE00000000000000000000")),
        Record(values = mapOf("purpose" to "salary november payout", "iban" to "DE00000000000000000001")),
        Record(values = mapOf("purpose" to "insurance premium renewal", "iban" to "DE00000000000000000002")),
    )
    println("Round-trip prediction comparison:")
    println("  %-40s  %-12s  %-12s  %s".format("purpose", "original", "restored", "match"))
    probes.forEach { probe ->
        val before = original.classify(record = probe)
        val after = restored.classify(record = probe)
        val purpose = probe["purpose"].toString().take(38)
        val match = if (before.label == after.label) "yes" else "NO — mismatch!"
        println("  %-40s  %-12s  %-12s  %s".format(purpose, before.label.value, after.label.value, match))
    }
}
