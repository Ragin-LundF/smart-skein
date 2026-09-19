package io.skein.classify.application

import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabelMetricsFactory
import io.skein.classify.domain.MultiLabelOutcome
import io.skein.classify.domain.MultiLabelPrediction
import io.skein.classify.domain.ScoredLabel
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class ThresholdOptimizerTest {

    private val optimizer = ThresholdOptimizer()
    private val rare = Label(value = "RARE")
    private val common = Label(value = "COMMON")

    private fun outcome(expected: Set<Label>, vararg scores: Pair<Label, Double>): MultiLabelOutcome {
        return MultiLabelOutcome(
            expected = expected,
            prediction = MultiLabelPrediction(
                ranked = scores.map { (label, probability) -> ScoredLabel(label = label, probability = probability) }
                    .sortedByDescending { scored -> scored.probability },
                threshold = 0.5,
            ),
        )
    }

    /**
     * RARE never scores above 0.4, so a global 0.5 silences it completely. Its scores still separate
     * its positives from its negatives cleanly, and a threshold fitted to the label finds that.
     */
    @Test
    internal fun `finds a low threshold for a label whose scores never reach the global cut`() {
        val outcomes = listOf(
            outcome(setOf(rare), rare to 0.35, common to 0.1),
            outcome(setOf(rare), rare to 0.31, common to 0.1),
            outcome(emptySet(), rare to 0.05, common to 0.1),
            outcome(emptySet(), rare to 0.02, common to 0.1),
        )

        val thresholds = optimizer.fit(outcomes = outcomes)

        assertTrue(
            actual = thresholds.of(label = rare) <= 0.31,
            message = "fitted ${thresholds.of(label = rare)}",
        )
        outcomes.forEach { outcome ->
            assertEquals(
                expected = outcome.expected.contains(element = rare),
                actual = thresholds.accepted(prediction = outcome.prediction).contains(element = rare),
            )
        }
    }

    @Test
    internal fun `finds a high threshold for a label that fires too readily`() {
        val outcomes = listOf(
            outcome(setOf(common), common to 0.95),
            outcome(setOf(common), common to 0.91),
            outcome(emptySet(), common to 0.70),
            outcome(emptySet(), common to 0.62),
        )

        val thresholds = optimizer.fit(outcomes = outcomes)

        assertTrue(actual = thresholds.of(label = common) > 0.70, message = "fitted ${thresholds.of(label = common)}")
    }

    /** Fitted thresholds must never score worse than the global cut they replace on the same data. */
    @Test
    internal fun `never scores below a uniform threshold on the data it was fitted to`() {
        val outcomes = listOf(
            outcome(setOf(rare), rare to 0.35, common to 0.80),
            outcome(setOf(rare, common), rare to 0.33, common to 0.91),
            outcome(setOf(common), rare to 0.04, common to 0.77),
            outcome(emptySet(), rare to 0.02, common to 0.20),
        )
        val thresholds = optimizer.fit(outcomes = outcomes)

        val tuned = outcomes.map { outcome ->
            MultiLabelOutcome(
                expected = outcome.expected,
                prediction = MultiLabelPrediction(
                    ranked = outcome.prediction.ranked.filter { scored ->
                        scored.label in thresholds.accepted(prediction = outcome.prediction)
                    },
                    threshold = 0.0,
                ),
            )
        }

        assertTrue(
            actual = MultiLabelMetricsFactory.from(outcomes = tuned).micro.f1 >=
                MultiLabelMetricsFactory.from(outcomes = outcomes).micro.f1,
        )
    }

    @Test
    internal fun `leaves a label with no positives on the fallback`() {
        val outcomes = listOf(
            outcome(setOf(common), common to 0.9, rare to 0.4),
            outcome(setOf(common), common to 0.8, rare to 0.3),
        )

        val thresholds = ThresholdOptimizer(fallback = 0.45).fit(outcomes = outcomes)

        assertEquals(expected = 0.45, actual = thresholds.of(label = rare))
        assertTrue(actual = rare !in thresholds.tunedLabels())
    }

    @Test
    internal fun `prefers the more conservative of two thresholds that score identically`() {
        // Every score above 0.2 is a true positive, so 0.3 and 0.9 both score a perfect F1.
        val outcomes = listOf(
            outcome(setOf(common), common to 0.9),
            outcome(emptySet(), common to 0.1),
        )

        val thresholds = optimizer.fit(outcomes = outcomes)

        assertEquals(expected = 0.9, actual = thresholds.of(label = common))
    }

    /**
     * Regression: an earlier implementation sampled a fixed number of candidate thresholds, which
     * could step straight over the only cut that separates the classes and settle for one that
     * admits a false positive. With several hundred distinct scores, the exact cut must still win.
     */
    @Test
    internal fun `finds the separating threshold in a corpus with many distinct scores`() {
        val outcomes = (0 until 400).map { index ->
            val positive = index < 201
            val probability = if (positive) 0.99 + index * 1e-5 else 1e-4 + index * 1e-5
            outcome(if (positive) setOf(common) else emptySet(), common to probability)
        }

        val thresholds = optimizer.fit(outcomes = outcomes)

        val accepted = outcomes.count { o -> common in thresholds.accepted(prediction = o.prediction) }
        assertEquals(expected = 201, actual = accepted)
        assertEquals(
            expected = 1.0,
            actual = MultiLabelMetricsFactory.from(
                outcomes = outcomes.map { o ->
                    MultiLabelOutcome(
                        expected = o.expected,
                        prediction = MultiLabelPrediction(
                            ranked = o.prediction.ranked.filter { scored ->
                                scored.label in thresholds.accepted(prediction = o.prediction)
                            },
                            threshold = 0.0,
                        ),
                    )
                },
            ).micro.f1,
            absoluteTolerance = 1e-12,
        )
    }

    @Test
    internal fun `rejects an empty outcome list or an out-of-range fallback`() {
        assertFailsWith<IllegalArgumentException> { optimizer.fit(outcomes = emptyList()) }
        assertFailsWith<IllegalArgumentException> { ThresholdOptimizer(fallback = 1.2) }
    }
}
