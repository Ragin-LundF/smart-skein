package io.skein.classify.domain

/**
 * Probability calibration applied to raw log-scores before the softmax: `softmax(z / temperature)`.
 *
 * A [temperature] above 1 softens an overconfident model — the common case here, because the Naive
 * Bayes log-likelihoods this library produces are unnormalized and saturate near 0 and 1. Below 1
 * sharpens. Temperature scaling is strictly rank-preserving, so it never changes which label wins,
 * only how confident the model claims to be.
 *
 * Fit it on data the model was **not** trained on, with
 * [io.skein.classify.application.TemperatureCalibrator]. Fitting on training data returns a
 * temperature near 1 and is a silent no-op.
 *
 * ponytail: one global scalar. Naive Bayes log-scores grow in magnitude with the number of non-zero
 * features, so a single temperature is a compromise across short and long records. Upgrade path:
 * per-label vector scaling (a temperature and bias per label), which adds fields here without
 * changing any call site.
 */
data class Calibration(val temperature: Double) {

    init {
        require(value = temperature > 0.0 && temperature.isFinite()) {
            "temperature must be a positive finite number, got $temperature"
        }
    }

    companion object {

        /** The identity calibration: the plain softmax this library applied before calibration existed. */
        val NONE: Calibration = Calibration(temperature = 1.0)
    }
}
