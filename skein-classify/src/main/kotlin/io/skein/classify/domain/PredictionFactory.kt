package io.skein.classify.domain

import kotlin.math.exp

/**
 * Builds a [Prediction] from per-label scores in log/logit space, using a numerically stable
 * softmax (subtracting the maximum before exponentiating). Shared by every classifier so the
 * probability calibration and ranking behave identically regardless of the underlying model.
 */
object PredictionFactory {

    fun fromLogScores(logScores: Map<Label, Double>): Prediction {
        require(value = logScores.isNotEmpty()) { "cannot build a prediction without scores" }
        val maxLog = logScores.values.max()
        val weights = logScores.mapValues { entry -> exp(x = entry.value - maxLog) }
        val total = weights.values.sum()
        val ranked = weights.entries
            .map { entry -> ScoredLabel(label = entry.key, probability = entry.value / total) }
            .sortedByDescending { scored -> scored.probability }
        val top = ranked.first()
        return Prediction(label = top.label, confidence = top.probability, alternatives = ranked)
    }
}
