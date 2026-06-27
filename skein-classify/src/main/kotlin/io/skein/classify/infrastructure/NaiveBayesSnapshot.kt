package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionFactory
import kotlin.math.ln

/**
 * Immutable trained state of [NaiveBayesClassifier]. Scoring reads it with no locks; learning
 * produces a new snapshot via [withObservation] (copy-on-write: only the touched label's structures
 * are rebuilt, the rest are shared by reference), which the classifier publishes atomically.
 */
internal class NaiveBayesSnapshot(
    private val labelDocumentCounts: Map<Label, Long>,
    private val featureSumsByLabel: Map<Label, Map<Int, Double>>,
    private val featureMassByLabel: Map<Label, Double>,
    private val totalDocuments: Long,
    private val vocabularySize: Int,
    private val smoothingAlpha: Double,
) {

    fun isTrained(): Boolean {
        return totalDocuments > 0L
    }

    fun labels(): Set<Label> {
        return labelDocumentCounts.keys
    }

    fun predict(features: FeatureVector): Prediction {
        val effectiveVocabulary = vocabularySize.coerceAtLeast(minimumValue = 1)
        val logScores = HashMap<Label, Double>()
        for (label in labelDocumentCounts.keys) {
            logScores[label] = logScoreFor(label = label, features = features, vocabularySize = effectiveVocabulary)
        }
        return PredictionFactory.fromLogScores(logScores = logScores)
    }

    /** Returns a new snapshot with one observation added; [newVocabularySize] is the distinct-feature count. */
    fun withObservation(features: FeatureVector, label: Label, newVocabularySize: Int): NaiveBayesSnapshot {
        val updatedCounts = HashMap(labelDocumentCounts)
        updatedCounts[label] = (labelDocumentCounts[label] ?: 0L) + 1L

        val updatedInner = HashMap(featureSumsByLabel[label] ?: emptyMap())
        var mass = featureMassByLabel[label] ?: 0.0
        for (position in features.indices.indices) {
            val index = features.indices[position]
            val value = features.values[position].toDouble()
            updatedInner[index] = (updatedInner[index] ?: 0.0) + value
            mass += value
        }

        val updatedSums = HashMap(featureSumsByLabel)
        updatedSums[label] = updatedInner
        val updatedMass = HashMap(featureMassByLabel)
        updatedMass[label] = mass

        return NaiveBayesSnapshot(
            labelDocumentCounts = updatedCounts,
            featureSumsByLabel = updatedSums,
            featureMassByLabel = updatedMass,
            totalDocuments = totalDocuments + 1L,
            vocabularySize = newVocabularySize,
            smoothingAlpha = smoothingAlpha,
        )
    }

    private fun logScoreFor(label: Label, features: FeatureVector, vocabularySize: Int): Double {
        val prior = ln(x = labelDocumentCounts.getValue(key = label).toDouble() / totalDocuments)
        val sums = featureSumsByLabel[label] ?: emptyMap()
        val denominator = (featureMassByLabel[label] ?: 0.0) + smoothingAlpha * vocabularySize
        var score = prior
        for (position in features.indices.indices) {
            val count = sums[features.indices[position]] ?: 0.0
            score += features.values[position].toDouble() * ln(x = (count + smoothingAlpha) / denominator)
        }
        return score
    }

    companion object {
        fun empty(smoothingAlpha: Double): NaiveBayesSnapshot {
            return NaiveBayesSnapshot(
                labelDocumentCounts = emptyMap(),
                featureSumsByLabel = emptyMap(),
                featureMassByLabel = emptyMap(),
                totalDocuments = 0L,
                vocabularySize = 0,
                smoothingAlpha = smoothingAlpha,
            )
        }
    }
}
