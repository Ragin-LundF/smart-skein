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
        val alternatives = ArrayList<ScoredLabel>(logScores.size)
        var total = 0.0
        for ((label, score) in logScores) {
            val w = exp(x = score - maxLog)
            alternatives.add(element = ScoredLabel(label = label, probability = w))
            total += w
        }
        for (i in alternatives.indices) {
            val s = alternatives[i]
            alternatives[i] = s.copy(probability = s.probability / total)
        }
        alternatives.sortByDescending { s -> s.probability }
        val top = alternatives.first()
        return Prediction(label = top.label, confidence = top.probability, alternatives = alternatives)
    }
}
