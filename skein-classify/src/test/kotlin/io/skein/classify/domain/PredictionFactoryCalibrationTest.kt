package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class PredictionFactoryCalibrationTest {

    private val tolerance = 1e-9

    private val scores = mapOf(
        Label(value = "A") to 12.0,
        Label(value = "B") to 8.0,
        Label(value = "C") to 3.0,
    )

    @Test
    internal fun `temperature one reproduces the uncalibrated overload exactly`() {
        val plain = PredictionFactory.fromLogScores(logScores = scores)
        val identity = PredictionFactory.fromLogScores(logScores = scores, calibration = Calibration.NONE)

        assertEquals(expected = plain.label, actual = identity.label)
        assertEquals(expected = plain.confidence, actual = identity.confidence, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `a higher temperature softens and a lower one sharpens`() {
        val base = PredictionFactory.fromLogScores(logScores = scores, calibration = Calibration.NONE)
        val soft = PredictionFactory.fromLogScores(logScores = scores, calibration = Calibration(temperature = 10.0))
        val sharp = PredictionFactory.fromLogScores(logScores = scores, calibration = Calibration(temperature = 0.1))

        assertTrue(actual = soft.confidence < base.confidence)
        assertTrue(actual = sharp.confidence > base.confidence)
    }

    @Test
    internal fun `calibration never reorders the labels`() {
        val baseline = PredictionFactory.fromLogScores(logScores = scores).alternatives.map { it.label }
        listOf(0.1, 0.5, 1.0, 10.0, 100.0).forEach { temperature ->
            val ranked = PredictionFactory
                .fromLogScores(logScores = scores, calibration = Calibration(temperature = temperature))
                .alternatives
                .map { scored -> scored.label }
            assertEquals(expected = baseline, actual = ranked, message = "reordered at T=$temperature")
        }
    }

    @Test
    internal fun `probabilities still sum to one at any temperature`() {
        listOf(0.05, 1.0, 50.0).forEach { temperature ->
            val total = PredictionFactory
                .fromLogScores(logScores = scores, calibration = Calibration(temperature = temperature))
                .alternatives
                .sumOf { scored -> scored.probability }
            assertEquals(expected = 1.0, actual = total, absoluteTolerance = 1e-9)
        }
    }

    @Test
    internal fun `extreme scores do not produce NaN or infinity`() {
        val extreme = mapOf(
            Label(value = "A") to -1e5,
            Label(value = "B") to -1.0000001e5,
        )
        val prediction = PredictionFactory
            .fromLogScores(logScores = extreme, calibration = Calibration(temperature = 2.0))

        assertTrue(actual = prediction.confidence.isFinite())
        prediction.alternatives.forEach { scored -> assertTrue(actual = scored.probability.isFinite()) }
    }
}
