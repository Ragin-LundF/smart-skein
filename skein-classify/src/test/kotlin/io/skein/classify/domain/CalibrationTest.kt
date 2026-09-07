package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class CalibrationTest {

    @Test
    internal fun `rejects a non-positive or non-finite temperature`() {
        listOf(0.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY).forEach { bad ->
            assertFailsWith<IllegalArgumentException> { Calibration(temperature = bad) }
        }
    }

    @Test
    internal fun `NONE is the identity temperature`() {
        assertEquals(expected = 1.0, actual = Calibration.NONE.temperature)
    }

    @Test
    internal fun `isConfident is inclusive at the threshold`() {
        val prediction = Prediction(
            label = Label(value = "A"),
            confidence = 0.75,
            alternatives = listOf(ScoredLabel(label = Label(value = "A"), probability = 0.75)),
        )
        assertTrue(actual = prediction.isConfident(minConfidence = 0.75))
        assertTrue(actual = prediction.isConfident(minConfidence = 0.0))
        assertTrue(actual = !prediction.isConfident(minConfidence = 0.76))
        assertFailsWith<IllegalArgumentException> { prediction.isConfident(minConfidence = 1.5) }
    }
}
