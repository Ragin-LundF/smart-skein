package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/**
 * The arithmetic is hand-computed in the comments below so the assertions pin the intended
 * averaging conventions rather than whatever the implementation happens to produce.
 */
internal class MultiLabelMetricsFactoryTest {

    private val tolerance = 1e-9

    private fun predictionOf(threshold: Double, vararg scores: Pair<String, Double>): MultiLabelPrediction {
        val ranked = scores
            .map { (label, probability) -> ScoredLabel(label = Label(value = label), probability = probability) }
            .sortedWith(
                comparator = compareByDescending<ScoredLabel> { scored -> scored.probability }
                    .thenBy { scored -> scored.label.value },
            )
        return MultiLabelPrediction(ranked = ranked, threshold = threshold)
    }

    private fun outcome(
        expected: Set<String>,
        threshold: Double = 0.5,
        vararg scores: Pair<String, Double>,
    ): MultiLabelOutcome {
        return MultiLabelOutcome(
            expected = expected.map { value -> Label(value = value) }.toSet(),
            prediction = predictionOf(threshold = threshold, scores = scores),
        )
    }

    /**
     * Three cases over labels A, B, C at threshold 0.5.
     *
     * 1. truth {A,B}, scores A .9 B .8 C .1 -> predicts {A,B}. tp 2, fp 0, fn 0.
     * 2. truth {A},   scores A .4 B .7 C .2 -> predicts {B}.   tp 0, fp 1, fn 1.
     * 3. truth {},    scores A .3 B .2 C .6 -> predicts {C}.   tp 0, fp 1, fn 0.
     *
     * Pooled: tp 2, fp 2, fn 1.
     */
    private fun mixedOutcomes(): List<MultiLabelOutcome> {
        return listOf(
            outcome(setOf("A", "B"), 0.5, "A" to 0.9, "B" to 0.8, "C" to 0.1),
            outcome(setOf("A"), 0.5, "A" to 0.4, "B" to 0.7, "C" to 0.2),
            outcome(emptySet(), 0.5, "A" to 0.3, "B" to 0.2, "C" to 0.6),
        )
    }

    @Test
    internal fun `micro average pools label decisions before dividing`() {
        val metrics = MultiLabelMetricsFactory.from(outcomes = mixedOutcomes())

        // precision 2/(2+2) = 0.5, recall 2/(2+1) = 2/3, f1 = 2*0.5*(2/3)/(0.5+2/3)
        assertEquals(expected = 0.5, actual = metrics.micro.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 2.0 / 3.0, actual = metrics.micro.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 4.0 / 7.0, actual = metrics.micro.f1, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `macro average weights every label equally including one never expected`() {
        val metrics = MultiLabelMetricsFactory.from(outcomes = mixedOutcomes())

        // A: tp 1 fp 0 fn 1 -> P 1.0  R 0.5  F1 2/3
        // B: tp 1 fp 1 fn 0 -> P 0.5  R 1.0  F1 2/3
        // C: tp 0 fp 1 fn 0 -> P 0.0  R 0.0  F1 0.0   (predicted, never true)
        assertEquals(expected = 0.5, actual = metrics.macro.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.5, actual = metrics.macro.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 4.0 / 9.0, actual = metrics.macro.f1, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `per-label metrics report support as the number of times the label was true`() {
        val metrics = MultiLabelMetricsFactory.from(outcomes = mixedOutcomes())

        val a = metrics.metricsFor(label = Label(value = "A"))!!
        assertEquals(expected = 1.0, actual = a.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.5, actual = a.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 2, actual = a.support)

        val b = metrics.metricsFor(label = Label(value = "B"))!!
        assertEquals(expected = 0.5, actual = b.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 1.0, actual = b.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 1, actual = b.support)
    }

    @Test
    internal fun `a label with no positives scores zero rather than NaN`() {
        val metrics = MultiLabelMetricsFactory.from(outcomes = mixedOutcomes())

        val c = metrics.metricsFor(label = Label(value = "C"))!!
        assertEquals(expected = 0.0, actual = c.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = c.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = c.f1, absoluteTolerance = tolerance)
        assertEquals(expected = 0, actual = c.support)
    }

    @Test
    internal fun `a label that never appears at all is absent rather than a zero row`() {
        val metrics = MultiLabelMetricsFactory.from(outcomes = mixedOutcomes())

        assertNull(actual = metrics.metricsFor(label = Label(value = "NEVER_SEEN")))
        assertEquals(expected = 3, actual = metrics.perLabel.size)
    }

    @Test
    internal fun `exact match counts only cases whose predicted set equals the expected set`() {
        val metrics = MultiLabelMetricsFactory.from(outcomes = mixedOutcomes())

        // Only case 1 matches exactly; case 2 is a swap and case 3 invents a label.
        assertEquals(expected = 1.0 / 3.0, actual = metrics.exactMatchRatio, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `an expected-empty case scores a false positive when the model invents a label`() {
        val onlyNegative = listOf(outcome(emptySet(), 0.5, "A" to 0.9, "B" to 0.1))

        val metrics = MultiLabelMetricsFactory.from(outcomes = onlyNegative)

        assertEquals(expected = 0.0, actual = metrics.micro.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = metrics.micro.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = metrics.exactMatchRatio, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `an expected-empty case that predicts nothing is an exact match with full coverage loss`() {
        val abstained = listOf(outcome(emptySet(), 0.5, "A" to 0.1, "B" to 0.2))

        val metrics = MultiLabelMetricsFactory.from(outcomes = abstained)

        assertEquals(expected = 1.0, actual = metrics.exactMatchRatio, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = metrics.coverage, absoluteTolerance = tolerance)
        assertEquals(expected = 0, actual = metrics.perLabel.size)
    }

    @Test
    internal fun `coverage is the fraction of cases that received any label`() {
        val outcomes = listOf(
            outcome(setOf("A"), 0.5, "A" to 0.9),
            outcome(setOf("A"), 0.5, "A" to 0.1),
            outcome(setOf("A"), 0.5, "A" to 0.7),
        )

        val metrics = MultiLabelMetricsFactory.from(outcomes = outcomes)

        assertEquals(expected = 2.0 / 3.0, actual = metrics.coverage, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `empty predictions across the board yield zero precision recall and coverage`() {
        val outcomes = listOf(
            outcome(setOf("A"), 0.9, "A" to 0.1, "B" to 0.2),
            outcome(setOf("B"), 0.9, "A" to 0.3, "B" to 0.4),
        )

        val metrics = MultiLabelMetricsFactory.from(outcomes = outcomes)

        assertEquals(expected = 0.0, actual = metrics.micro.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = metrics.micro.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = metrics.micro.f1, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = metrics.coverage, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `recall at k ignores the threshold and grows with depth`() {
        val outcomes = mixedOutcomes()

        // k=1: case 1 finds A of {A,B}; case 2's top is B, so it finds none of {A}. 1 of 3.
        assertEquals(
            expected = 1.0 / 3.0,
            actual = MultiLabelMetricsFactory.recallAtK(outcomes = outcomes, count = 1),
            absoluteTolerance = tolerance,
        )
        // k=2: case 1 finds A and B; case 2's second candidate is A. 3 of 3.
        assertEquals(
            expected = 1.0,
            actual = MultiLabelMetricsFactory.recallAtK(outcomes = outcomes, count = 2),
            absoluteTolerance = tolerance,
        )
    }

    @Test
    internal fun `recall at k is zero when no case expects a label`() {
        val outcomes = listOf(outcome(emptySet(), 0.5, "A" to 0.9))

        assertEquals(
            expected = 0.0,
            actual = MultiLabelMetricsFactory.recallAtK(outcomes = outcomes, count = 3),
            absoluteTolerance = tolerance,
        )
    }

    @Test
    internal fun `rejects an empty outcome list`() {
        assertFailsWith<IllegalArgumentException> { MultiLabelMetricsFactory.from(outcomes = emptyList()) }
    }

    @Test
    internal fun `rejects a non-positive rank depth`() {
        assertFailsWith<IllegalArgumentException> {
            MultiLabelMetricsFactory.recallAtK(outcomes = mixedOutcomes(), count = 0)
        }
    }
}
