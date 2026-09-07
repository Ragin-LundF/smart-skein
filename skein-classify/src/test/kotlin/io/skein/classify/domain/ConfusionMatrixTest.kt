package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class ConfusionMatrixTest {

    private fun outcome(expected: String, predicted: String): PredictionOutcome {
        return PredictionOutcome(
            expected = Label(value = expected),
            prediction = Prediction(
                label = Label(value = predicted),
                confidence = 1.0,
                alternatives = listOf(ScoredLabel(label = Label(value = predicted), probability = 1.0)),
            ),
        )
    }

    @Test
    internal fun `tallies every expected-predicted pair`() {
        val matrix = ConfusionMatrix.of(
            outcomes = listOf(
                outcome(expected = "A", predicted = "A"),
                outcome(expected = "A", predicted = "B"),
                outcome(expected = "B", predicted = "B"),
                outcome(expected = "C", predicted = "A"),
            ),
        )
        assertEquals(expected = 1, actual = matrix.count(expected = Label(value = "A"), predicted = Label(value = "A")))
        assertEquals(expected = 1, actual = matrix.count(expected = Label(value = "A"), predicted = Label(value = "B")))
        assertEquals(expected = 0, actual = matrix.count(expected = Label(value = "B"), predicted = Label(value = "A")))
        assertEquals(expected = 1, actual = matrix.count(expected = Label(value = "C"), predicted = Label(value = "A")))
        assertEquals(expected = 4, actual = matrix.total())
    }

    @Test
    internal fun `row total is support and column total is prediction count`() {
        val matrix = ConfusionMatrix.of(
            outcomes = listOf(
                outcome(expected = "A", predicted = "A"),
                outcome(expected = "A", predicted = "B"),
                outcome(expected = "B", predicted = "B"),
            ),
        )
        assertEquals(expected = 2, actual = matrix.rowTotal(label = Label(value = "A")))
        assertEquals(expected = 1, actual = matrix.rowTotal(label = Label(value = "B")))
        assertEquals(expected = 1, actual = matrix.columnTotal(label = Label(value = "A")))
        assertEquals(expected = 2, actual = matrix.columnTotal(label = Label(value = "B")))
        assertEquals(expected = 1, actual = matrix.correct(label = Label(value = "A")))
    }

    @Test
    internal fun `labels are the sorted union of expected and predicted`() {
        val matrix = ConfusionMatrix.of(
            outcomes = listOf(
                outcome(expected = "zebra", predicted = "alpha"),
                outcome(expected = "mid", predicted = "mid"),
            ),
        )
        assertEquals(
            expected = listOf(Label(value = "alpha"), Label(value = "mid"), Label(value = "zebra")),
            actual = matrix.labels,
        )
    }

    @Test
    internal fun `a label that is only ever predicted has zero support`() {
        val matrix = ConfusionMatrix.of(outcomes = listOf(outcome(expected = "A", predicted = "ghost")))
        assertTrue(actual = Label(value = "ghost") in matrix.labels)
        assertEquals(expected = 0, actual = matrix.rowTotal(label = Label(value = "ghost")))
        assertEquals(expected = 1, actual = matrix.columnTotal(label = Label(value = "ghost")))
    }

    @Test
    internal fun `rejects an unknown label`() {
        val matrix = ConfusionMatrix.of(outcomes = listOf(outcome(expected = "A", predicted = "A")))
        assertFailsWith<IllegalArgumentException> {
            matrix.count(expected = Label(value = "A"), predicted = Label(value = "nope"))
        }
    }

    @Test
    internal fun `top confusions returns off-diagonal cells descending and honours the limit`() {
        val outcomes = List(size = 5) { outcome(expected = "A", predicted = "B") } +
            List(size = 3) { outcome(expected = "B", predicted = "C") } +
            List(size = 7) { outcome(expected = "A", predicted = "A") }
        val confusions = ConfusionMatrix.of(outcomes = outcomes).topConfusions(limit = 10)

        assertEquals(expected = 2, actual = confusions.size)
        assertEquals(
            expected = Triple(first = Label(value = "A"), second = Label(value = "B"), third = 5),
            actual = confusions[0],
        )
        assertEquals(
            expected = Triple(first = Label(value = "B"), second = Label(value = "C"), third = 3),
            actual = confusions[1],
        )
        assertEquals(expected = 1, actual = ConfusionMatrix.of(outcomes = outcomes).topConfusions(limit = 1).size)
    }

    @Test
    internal fun `rejects a non-positive confusion limit`() {
        val matrix = ConfusionMatrix.of(outcomes = listOf(outcome(expected = "A", predicted = "A")))
        assertFailsWith<IllegalArgumentException> { matrix.topConfusions(limit = 0) }
    }
}
