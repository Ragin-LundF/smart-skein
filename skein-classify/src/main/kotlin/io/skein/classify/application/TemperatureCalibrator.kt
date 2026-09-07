package io.skein.classify.application

import io.skein.classify.domain.Calibration
import io.skein.classify.domain.CalibrationSample
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.spi.Classifier
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/** Smallest inverse temperature considered, i.e. the softest calibration reachable. */
private const val BETA_MIN = 1e-3

/** Largest inverse temperature considered, i.e. the sharpest calibration reachable. */
private const val BETA_MAX = 1e3

/** Golden-section shrink factor, `1 / phi`. */
private const val GOLDEN_RATIO_INVERSE = 0.618033988749895

/** Hard iteration cap; the bracket normally closes well before this. */
private const val MAX_ITERATIONS = 200

/** Bracket width at which the search stops. */
private const val BRACKET_TOLERANCE = 1e-9

/** A distribution needs at least two labels before a temperature means anything. */
private const val MIN_LABELS_FOR_FITTING = 2

/**
 * Fits a [Calibration] by minimizing the negative log-likelihood of held-out data.
 *
 * **The data must be held out.** Scores on rows the model trained on already look well calibrated,
 * so fitting there returns a temperature near 1 and silently does nothing.
 *
 * The search runs over the inverse temperature `beta = 1 / T`, where the objective
 * `NLL(beta) = mean over samples of [ logSumExp(beta * z) - beta * z_true ]` is convex: its second
 * derivative is the variance of the scores under the model's own distribution, which cannot be
 * negative. It is *not* convex in `T` itself, which is why the search is parameterized this way.
 * Golden-section search then needs no derivatives, cannot diverge, and is deterministic.
 *
 * ponytail: golden section costs about 200 evaluations where a safeguarded Newton step would need
 * roughly five. Ceiling: only matters if fitting moves onto a hot path, and it is an offline,
 * `O(samples * labels)` operation. Upgrade path: Newton's method on the analytic gradient, guarded
 * against a near-zero Hessian.
 */
class TemperatureCalibrator {

    /**
     * Fits a temperature to [samples]. Returns [Calibration.NONE] when the samples carry fewer than
     * two labels, since no temperature is identifiable from a single-label distribution.
     */
    fun fit(samples: List<CalibrationSample>): Calibration {
        require(value = samples.isNotEmpty()) { "cannot fit a calibration without samples" }
        samples.forEach { sample ->
            require(value = sample.logScores.containsKey(key = sample.trueLabel)) {
                "sample for '${sample.trueLabel.value}' has no score for its own true label"
            }
        }
        if (samples.none { sample -> sample.logScores.size >= MIN_LABELS_FOR_FITTING }) {
            return Calibration(temperature = Calibration.NONE.temperature)
        }
        return Calibration(temperature = 1.0 / minimizingBeta(samples = samples))
    }

    /**
     * Scores [heldOut] with [classifier] and fits a temperature to the result. [heldOut] must not
     * be part of what [classifier] learned from.
     */
    fun fit(heldOut: List<LabeledFeatures>, classifier: Classifier): Calibration {
        require(value = heldOut.isNotEmpty()) { "cannot fit a calibration without held-out data" }
        val samples = heldOut.map { observation ->
            CalibrationSample(
                trueLabel = observation.label,
                logScores = classifier.logScores(features = observation.features),
            )
        }
        return fit(samples = samples)
    }

    /** Golden-section minimization of [negativeLogLikelihood] over the inverse temperature. */
    private fun minimizingBeta(samples: List<CalibrationSample>): Double {
        var low = BETA_MIN
        var high = BETA_MAX
        var probeLow = high - GOLDEN_RATIO_INVERSE * (high - low)
        var probeHigh = low + GOLDEN_RATIO_INVERSE * (high - low)
        var valueLow = negativeLogLikelihood(samples = samples, beta = probeLow)
        var valueHigh = negativeLogLikelihood(samples = samples, beta = probeHigh)

        var iteration = 0
        while (iteration < MAX_ITERATIONS && abs(x = high - low) > BRACKET_TOLERANCE) {
            if (valueLow < valueHigh) {
                high = probeHigh
                probeHigh = probeLow
                valueHigh = valueLow
                probeLow = high - GOLDEN_RATIO_INVERSE * (high - low)
                valueLow = negativeLogLikelihood(samples = samples, beta = probeLow)
            } else {
                low = probeLow
                probeLow = probeHigh
                valueLow = valueHigh
                probeHigh = low + GOLDEN_RATIO_INVERSE * (high - low)
                valueHigh = negativeLogLikelihood(samples = samples, beta = probeHigh)
            }
            iteration += 1
        }
        return (low + high) / 2.0
    }

    /**
     * Mean negative log-likelihood of the true labels once every score is scaled by [beta].
     *
     * Computed as `logSumExp(beta * z) - beta * z_true`, which is invariant to a per-sample additive
     * shift of the scores — so the arbitrary offset carried by unnormalized log-likelihoods does not
     * affect the fit.
     */
    private fun negativeLogLikelihood(samples: List<CalibrationSample>, beta: Double): Double {
        var total = 0.0
        for (sample in samples) {
            val scores = sample.logScores.values
            val maximum = scores.max() * beta
            var sumOfExponentials = 0.0
            for (score in scores) {
                sumOfExponentials += exp(x = score * beta - maximum)
            }
            val logSumExp = maximum + ln(x = sumOfExponentials)
            total += logSumExp - sample.logScores.getValue(key = sample.trueLabel) * beta
        }
        return total / samples.size
    }
}
