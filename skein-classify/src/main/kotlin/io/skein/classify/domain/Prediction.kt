package io.skein.classify.domain

/**
 * Result of classifying a record: the winning [label], its [confidence] (the top probability),
 * and the full ranked list of [alternatives] (highest probability first, including the winner).
 */
data class Prediction(
    val label: Label,
    val confidence: Double,
    val alternatives: List<ScoredLabel>,
) {

    /**
     * Whether [confidence] reaches [minConfidence] — the abstain test. A threshold of `0.0` always
     * accepts, which is the behaviour of a caller that does not abstain at all.
     *
     * Only meaningful against a calibrated prediction: raw Naive Bayes confidences sit close to 0
     * and 1, and no threshold between them separates anything useful. See
     * [io.skein.classify.domain.Calibration].
     */
    fun isConfident(minConfidence: Double): Boolean {
        require(value = minConfidence in 0.0..1.0) { "minConfidence must be between 0.0 and 1.0" }
        return confidence >= minConfidence
    }
}
