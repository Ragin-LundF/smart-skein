package io.skein.examples.logwatch

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.ClassifierKindEnum
import io.skein.classify.application.ModelStore
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.InMemoryFeatureStore
import io.skein.classify.infrastructure.NaiveBayesClassifier
import java.nio.file.Path

// Schema for a single log line: free-form message text + categorical level.
val LOG_SCHEMA: Schema = Schema.define {
    text("message")
    categorical("level")
    label("anomaly_type")
}

const val NORMAL_LABEL = "NORMAL"

// Fixed keys so a saved model can be reloaded and produce the same feature hashes.
private val HASHING_CONFIG = HashingConfig(key0 = 0x6c6f67776174636bL, key1 = 0x616e6f6d616c7953L)

/**
 * Trains a [ClassificationService] from a CSV with columns: anomaly_type, level, keyword.
 * Each row becomes one labeled training record; the keyword is the message text.
 *
 * Returns the trained service and the backing feature store (needed to save the model).
 */
fun trainFromRulesCsv(csv: String): Pair<ClassificationService, InMemoryFeatureStore> {
    val store = InMemoryFeatureStore()
    val service = ClassificationService(
        schema = LOG_SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HASHING_CONFIG,
        classifier = NaiveBayesClassifier(),
        featureStore = store,
    )
    val records = parseRulesCsv(csv).map { row -> Record(values = row) }
    service.learnAll(records = records)
    return service to store
}

/** Saves the model so it can be loaded later without retraining. */
fun saveModel(store: InMemoryFeatureStore, path: Path) {
    ModelStore.save(
        path = path,
        schema = LOG_SCHEMA,
        classifier = ClassifierKindEnum.NAIVE_BAYES,
        hashingConfig = HASHING_CONFIG,
        observations = store.all(),
    )
}

/** Loads a previously saved model and rebuilds the Naive Bayes classifier from stored observations. */
fun loadModel(path: Path): ClassificationService {
    val model = ModelStore.load(path = path)
    val store = InMemoryFeatureStore()
    store.addAll(observations = model.observations)
    val service = ClassificationService(
        schema = model.schema,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = model.hashingConfig,
        classifier = NaiveBayesClassifier(),
        featureStore = store,
    )
    service.retrain(epochs = 1)
    return service
}

/** Classifies a single raw log line. Returns null if no log level can be found in the line. */
fun ClassificationService.classifyLine(line: String): LogPrediction? {
    val (level, message) = parseLine(line) ?: return null
    val result = classify(record = Record(values = mapOf("message" to message, "level" to level)))
    return LogPrediction(line = line, level = level, label = result.label.value, confidence = result.confidence)
}

data class LogPrediction(val line: String, val level: String, val label: String, val confidence: Double)

// --- internal helpers ---

private val LOG_LEVELS = listOf("ERROR", "WARN", "INFO", "DEBUG", "TRACE")

fun parseLine(line: String): Pair<String, String>? {
    val level = LOG_LEVELS.firstOrNull { level ->
        line.split(Regex("[\\s\\[\\]:|]+")).any { token -> token.equals(level, ignoreCase = true) }
    } ?: return null
    val message = line.substringAfter(" - ", missingDelimiterValue = line).trim()
    return level to message
}

private fun parseRulesCsv(csv: String): List<Map<String, Any?>> {
    val lines = csv.lines().filter { it.isNotBlank() }
    val header = lines.first().split(",").map { it.trim() }
    return lines.drop(1).map { line ->
        val values = line.split(",", limit = header.size).map { it.trim() }
        header.zip(values).toMap()
    }
}
