package io.skein.classify.domain

import kotlin.math.exp

/**
 * Builds a [Prediction] from per-label scores in log/logit space, using a numerically stable
 * softmax (subtracting the maximum before exponentiating). Shared by every classifier so the
 * probability calibration and ranking behave identically regardless of the underlying model.
 */
object PredictionFactory {

    /**
     * Uncalibrated softmax over [logScores].
     *
     * Kept as its own overload rather than a defaulted parameter on the two-argument form: a default
     * argument would replace this method's JVM signature with a synthetic bridge, breaking every
     * call site compiled against an earlier release.
     */
    fun fromLogScores(logScores: Map<Label, Double>): Prediction {
        return fromLogScores(logScores = logScores, calibration = Calibration.NONE)
    }

    /**
     * Softmax over `logScores / calibration.temperature`.
     *
     * Dividing by a positive temperature cannot reorder the scores, so subtracting the maximum
     * remains a valid stabilization and the winning label is unchanged.
     */
    fun fromLogScores(logScores: Map<Label, Double>, calibration: Calibration): Prediction {
        require(value = logScores.isNotEmpty()) { "cannot build a prediction without scores" }
        val inverseTemperature = 1.0 / calibration.temperature
        val maxLog = logScores.values.max()
        val alternatives = ArrayList<ScoredLabel>(logScores.size)
        var total = 0.0
        for ((label, score) in logScores) {
            val w = exp(x = (score - maxLog) * inverseTemperature)
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
