package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionFactory

/**
 * Immutable trained state of [LogisticRegressionSgdClassifier] — per-label sparse weights and bias.
 * Scoring reads it lock-free; the classifier publishes a fresh snapshot after each SGD step.
 */
internal class LogisticRegressionSnapshot(
    private val weightsByLabel: Map<Label, Map<Int, Double>>,
    private val biasByLabel: Map<Label, Double>,
) {

    fun isTrained(): Boolean {
        return weightsByLabel.isNotEmpty()
    }

    fun labels(): Set<Label> {
        return weightsByLabel.keys
    }

    fun predict(features: FeatureVector): Prediction {
        val logits = HashMap<Label, Double>()
        for (label in weightsByLabel.keys) {
            logits[label] = scoreFor(label = label, features = features)
        }
        return PredictionFactory.fromLogScores(logScores = logits)
    }

    private fun scoreFor(label: Label, features: FeatureVector): Double {
        val weights = weightsByLabel.getValue(key = label)
        var score = biasByLabel[label] ?: 0.0
        for (position in features.indices.indices) {
            score += (weights[features.indices[position]] ?: 0.0) * features.values[position].toDouble()
        }
        return score
    }

    companion object {
        fun empty(): LogisticRegressionSnapshot {
            return LogisticRegressionSnapshot(weightsByLabel = emptyMap(), biasByLabel = emptyMap())
        }

        /** Deep-copies the mutable training maps into an immutable snapshot. */
        fun of(
            weightsByLabel: Map<Label, Map<Int, Double>>,
            biasByLabel: Map<Label, Double>,
        ): LogisticRegressionSnapshot {
            val copiedWeights = HashMap<Label, Map<Int, Double>>()
            for ((label, inner) in weightsByLabel) {
                copiedWeights[label] = HashMap(inner)
            }
            return LogisticRegressionSnapshot(weightsByLabel = copiedWeights, biasByLabel = HashMap(biasByLabel))
        }
    }
}
