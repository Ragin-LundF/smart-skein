package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class MultiLabelPredictionTest {

    private fun prediction(threshold: Double, vararg scores: Pair<String, Double>): MultiLabelPrediction {
        val ranked = scores
            .map { (label, probability) -> ScoredLabel(label = Label(value = label), probability = probability) }
        return MultiLabelPrediction(ranked = ranked, threshold = threshold)
    }

    @Test
    internal fun `labels accepts every score at or above the threshold`() {
        val prediction = prediction(0.5, "A" to 0.9, "B" to 0.5, "C" to 0.49)

        assertEquals(
            expected = setOf(Label(value = "A"), Label(value = "B")),
            actual = prediction.labels(),
        )
    }

    @Test
    internal fun `labels is empty when the model is confident about none of them`() {
        val prediction = prediction(0.5, "A" to 0.4, "B" to 0.1)

        assertTrue(actual = prediction.labels().isEmpty())
    }

    @Test
    internal fun `a threshold of zero accepts every label`() {
        val prediction = prediction(0.0, "A" to 0.4, "B" to 0.0)

        assertEquals(expected = 2, actual = prediction.labels().size)
    }

    @Test
    internal fun `topK returns the highest scores regardless of the threshold`() {
        val prediction = prediction(0.9, "A" to 0.8, "B" to 0.4, "C" to 0.1)

        val top = prediction.topK(count = 2)

        assertTrue(actual = prediction.labels().isEmpty())
        assertEquals(expected = listOf(Label(value = "A"), Label(value = "B")), actual = top.map { it.label })
    }

    @Test
    internal fun `topK returns every label when asked for more than the model knows`() {
        val prediction = prediction(0.5, "A" to 0.8, "B" to 0.4)

        assertEquals(expected = 2, actual = prediction.topK(count = 10).size)
    }

    @Test
    internal fun `at re-reads the same scores under a different threshold`() {
        val prediction = prediction(0.5, "A" to 0.9, "B" to 0.6, "C" to 0.2)

        val stricter = prediction.at(threshold = 0.8)

        assertEquals(expected = setOf(Label(value = "A")), actual = stricter.labels())
        assertEquals(expected = prediction.ranked, actual = stricter.ranked)
    }

    @Test
    internal fun `probabilityOf reports zero for a label the model does not know`() {
        val prediction = prediction(0.5, "A" to 0.9)

        assertEquals(expected = 0.9, actual = prediction.probabilityOf(label = Label(value = "A")))
        assertEquals(expected = 0.0, actual = prediction.probabilityOf(label = Label(value = "UNKNOWN")))
    }

    @Test
    internal fun `rejects a threshold outside zero to one`() {
        assertFailsWith<IllegalArgumentException> { prediction(1.5, "A" to 0.9) }
        assertFailsWith<IllegalArgumentException> { prediction(-0.1, "A" to 0.9) }
    }

    @Test
    internal fun `rejects a non-positive topK depth`() {
        assertFailsWith<IllegalArgumentException> { prediction(0.5, "A" to 0.9).topK(count = 0) }
    }
}
