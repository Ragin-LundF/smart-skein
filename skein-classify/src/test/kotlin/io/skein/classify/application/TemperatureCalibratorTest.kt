package io.skein.classify.application

import io.skein.classify.domain.Calibration
import io.skein.classify.domain.CalibrationSample
import io.skein.classify.domain.EvaluationReportFactory
import io.skein.classify.domain.Label
import io.skein.classify.domain.PredictionFactory
import io.skein.classify.domain.PredictionOutcome
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class TemperatureCalibratorTest {

    private val labelA = Label(value = "A")
    private val labelB = Label(value = "B")

    /**
     * Overconfident samples: the score gap is large (so the softmax saturates), but the top label is
     * only correct about 70% of the time. That is exactly the Naive Bayes failure mode.
     */
    private fun overconfidentSamples(count: Int = 400): List<CalibrationSample> {
        val random = Random(seed = 99L)
        return List(size = count) {
            val topIsCorrect = random.nextDouble() < 0.7
            val scores = mapOf(labelA to 9.0, labelB to 0.0)
            CalibrationSample(trueLabel = if (topIsCorrect) labelA else labelB, logScores = scores)
        }
    }

    private fun negativeLogLikelihood(samples: List<CalibrationSample>, calibration: Calibration): Double {
        return samples.sumOf { sample ->
            val prediction = PredictionFactory.fromLogScores(
                logScores = sample.logScores,
                calibration = calibration,
            )
            val probability = prediction.alternatives
                .first { scored -> scored.label == sample.trueLabel }
                .probability
            -ln(x = probability)
        } / samples.size
    }

    private fun expectedCalibrationError(samples: List<CalibrationSample>, calibration: Calibration): Double {
        val outcomes = samples.map { sample ->
            PredictionOutcome(
                expected = sample.trueLabel,
                prediction = PredictionFactory.fromLogScores(
                    logScores = sample.logScores,
                    calibration = calibration,
                ),
            )
        }
        return EvaluationReportFactory.from(outcomes = outcomes).expectedCalibrationError
    }

    @Test
    internal fun `fitting lowers held-out negative log-likelihood`() {
        val samples = overconfidentSamples()
        val fitted = TemperatureCalibrator().fit(samples = samples)

        val before = negativeLogLikelihood(samples = samples, calibration = Calibration.NONE)
        val after = negativeLogLikelihood(samples = samples, calibration = fitted)

        assertTrue(actual = after < before, message = "NLL rose from $before to $after")
        assertTrue(actual = fitted.temperature > 1.0, message = "expected softening, got T=${fitted.temperature}")
    }

    @Test
    internal fun `fitting lowers the expected calibration error`() {
        val samples = overconfidentSamples()
        val fitted = TemperatureCalibrator().fit(samples = samples)

        val before = expectedCalibrationError(samples = samples, calibration = Calibration.NONE)
        val after = expectedCalibrationError(samples = samples, calibration = fitted)

        assertTrue(actual = after < before, message = "ECE rose from $before to $after")
    }

    @Test
    internal fun `fitting never changes which label wins`() {
        val samples = overconfidentSamples()
        val fitted = TemperatureCalibrator().fit(samples = samples)

        val before = samples.count { sample ->
            PredictionFactory.fromLogScores(sample.logScores, Calibration.NONE).label == sample.trueLabel
        }
        val after = samples.count { sample ->
            PredictionFactory.fromLogScores(sample.logScores, fitted).label == sample.trueLabel
        }
        assertEquals(expected = before, actual = after)
    }

    @Test
    internal fun `already-calibrated scores fit a temperature near one`() {
        // Build samples whose softmax probability genuinely matches the outcome frequency.
        val random = Random(seed = 7L)
        val samples = List(size = 600) {
            val gap = random.nextDouble() * 2.0
            val probabilityOfA = exp(x = gap) / (exp(x = gap) + 1.0)
            CalibrationSample(
                trueLabel = if (random.nextDouble() < probabilityOfA) labelA else labelB,
                logScores = mapOf(labelA to gap, labelB to 0.0),
            )
        }
        val fitted = TemperatureCalibrator().fit(samples = samples)

        assertTrue(
            actual = fitted.temperature in 0.7..1.4,
            message = "expected a temperature near 1, got ${fitted.temperature}",
        )
    }

    @Test
    internal fun `the fit is deterministic`() {
        val samples = overconfidentSamples()
        assertEquals(
            expected = TemperatureCalibrator().fit(samples = samples).temperature,
            actual = TemperatureCalibrator().fit(samples = samples).temperature,
        )
    }

    @Test
    internal fun `the fitted temperature is a local minimum of the objective`() {
        val samples = overconfidentSamples()
        val fitted = TemperatureCalibrator().fit(samples = samples)
        val best = negativeLogLikelihood(samples = samples, calibration = fitted)

        listOf(0.6, 0.8, 1.25, 1.6).forEach { factor ->
            val nudged = Calibration(temperature = fitted.temperature * factor)
            assertTrue(
                actual = best <= negativeLogLikelihood(samples = samples, calibration = nudged),
                message = "T=${fitted.temperature} beaten by T=${nudged.temperature}",
            )
        }
    }

    @Test
    internal fun `single-label scores are not identifiable and fit to NONE`() {
        val samples = listOf(CalibrationSample(trueLabel = labelA, logScores = mapOf(labelA to 3.0)))
        assertEquals(expected = Calibration.NONE, actual = TemperatureCalibrator().fit(samples = samples))
    }

    @Test
    internal fun `rejects empty input and a sample missing its own true label`() {
        assertFailsWith<IllegalArgumentException> { TemperatureCalibrator().fit(samples = emptyList()) }

        val failure = assertFailsWith<IllegalArgumentException> {
            TemperatureCalibrator().fit(
                samples = listOf(CalibrationSample(trueLabel = labelA, logScores = mapOf(labelB to 1.0))),
            )
        }
        assertTrue(actual = failure.message.orEmpty().contains(other = "true label"))
    }
}
