package io.skein.classify.infrastructure

import io.skein.classify.domain.Explanation
import io.skein.classify.domain.FeatureContribution
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionFactory
import kotlin.math.abs
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

    private val featureLookupByLabel: Map<Label, IntDoubleHashMap>

    init {
        val lookupMap = HashMap<Label, IntDoubleHashMap>(featureSumsByLabel.size)
        for ((label, sums) in featureSumsByLabel) {
            val lookup = IntDoubleHashMap(initialCapacity = sums.size * 2)
            for ((k, v) in sums) lookup.put(key = k, value = v)
            lookupMap[label] = lookup
        }
        featureLookupByLabel = lookupMap
    }

    fun isTrained(): Boolean {
        return totalDocuments > 0L
    }

    fun labels(): Set<Label> {
        return labelDocumentCounts.keys
    }

    fun predict(features: FeatureVector): Prediction {
        return PredictionFactory.fromLogScores(logScores = logScores(features = features))
    }

    /** Raw per-label log-likelihoods, before any softmax. */
    fun logScores(features: FeatureVector): Map<Label, Double> {
        val effectiveVocabulary = vocabularySize.coerceAtLeast(minimumValue = 1)
        val logScores = HashMap<Label, Double>()
        for (label in labelDocumentCounts.keys) {
            logScores[label] = logScoreFor(label = label, features = features, vocabularySize = effectiveVocabulary)
        }
        return logScores
    }

    /**
     * Decomposes [label]'s score into per-feature contributions, each centered on the mean across
     * all labels.
     *
     * Centering is essential rather than cosmetic here: every Naive Bayes term is a log-probability
     * and therefore negative, so ranking the raw terms would rank by "least negative" and surface
     * whichever n-grams are commonest overall. Subtracting the mean turns each term into a log-odds
     * ratio against the average label, so a feature equally likely under every label contributes
     * exactly zero.
     */
    fun explain(features: FeatureVector, label: Label, limit: Int): Explanation {
        val effectiveVocabulary = vocabularySize.coerceAtLeast(minimumValue = 1)
        val labels = labelDocumentCounts.keys.toList()
        val denominators = labels.associateWith { candidate ->
            (featureMassByLabel[candidate] ?: 0.0) + smoothingAlpha * effectiveVocabulary
        }
        val base = ln(x = labelDocumentCounts.getValue(key = label).toDouble() / totalDocuments) -
            labels.sumOf { candidate ->
                ln(x = labelDocumentCounts.getValue(key = candidate).toDouble() / totalDocuments)
            } / labels.size

        var total = base
        val contributions = ArrayList<FeatureContribution>(features.indices.size)
        for (position in features.indices.indices) {
            val index = features.indices[position]
            val value = features.values[position].toDouble()
            val termFor = { candidate: Label ->
                val count = featureLookupByLabel[candidate]?.get(key = index) ?: 0.0
                value * ln(x = (count + smoothingAlpha) / denominators.getValue(key = candidate))
            }
            val centered = termFor(label) - labels.sumOf { candidate -> termFor(candidate) } / labels.size
            total += centered
            contributions.add(
                element = FeatureContribution(
                    featureIndex = index,
                    featureValue = features.values[position],
                    contribution = centered,
                ),
            )
        }
        contributions.sortByDescending { contribution -> abs(x = contribution.contribution) }
        return Explanation(
            label = label,
            probability = predict(features = features).alternatives
                .first { scored -> scored.label == label }
                .probability,
            base = base,
            total = total,
            contributions = contributions.take(n = limit),
        )
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
        val lookup = featureLookupByLabel[label]
        val denominator = (featureMassByLabel[label] ?: 0.0) + smoothingAlpha * vocabularySize
        var score = prior
        for (position in features.indices.indices) {
            val count = lookup?.get(key = features.indices[position]) ?: 0.0
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
