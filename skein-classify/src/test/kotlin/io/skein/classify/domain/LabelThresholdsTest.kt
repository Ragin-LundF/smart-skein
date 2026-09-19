package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class LabelThresholdsTest {

    private val rare = Label(value = "RARE")
    private val common = Label(value = "COMMON")

    private fun prediction(vararg scores: Pair<Label, Double>): MultiLabelPrediction {
        return MultiLabelPrediction(
            ranked = scores.map { (label, probability) -> ScoredLabel(label = label, probability = probability) }
                .sortedByDescending { scored -> scored.probability },
            threshold = 0.5,
        )
    }

    /**
     * The whole reason per-label thresholds exist: a poorly supported label produces timid
     * probabilities that a global cut silences entirely, even where it is clearly the best answer
     * available for that record.
     */
    @Test
    internal fun `a low threshold lets an under-confident label through where a global cut would not`() {
        val thresholds = LabelThresholds(byLabel = mapOf(rare to 0.2), fallback = 0.5)
        val scores = prediction(rare to 0.31, common to 0.72)

        assertEquals(expected = setOf(common), actual = scores.labels())
        assertEquals(expected = setOf(rare, common), actual = thresholds.accepted(prediction = scores))
    }

    @Test
    internal fun `a high threshold holds an over-firing label back`() {
        val thresholds = LabelThresholds(byLabel = mapOf(common to 0.9), fallback = 0.5)

        assertEquals(
            expected = emptySet(),
            actual = thresholds.accepted(prediction = prediction(common to 0.72)),
        )
    }

    @Test
    internal fun `an untuned label falls back`() {
        val thresholds = LabelThresholds(byLabel = mapOf(rare to 0.2), fallback = 0.6)

        assertEquals(expected = 0.6, actual = thresholds.of(label = common))
        assertEquals(expected = 0.2, actual = thresholds.of(label = rare))
        assertEquals(expected = setOf(rare), actual = thresholds.tunedLabels())
    }

    @Test
    internal fun `a uniform set behaves exactly like a global cut`() {
        val thresholds = LabelThresholds.uniform(threshold = 0.5)
        val scores = prediction(rare to 0.31, common to 0.72)

        assertEquals(expected = scores.labels(), actual = thresholds.accepted(prediction = scores))
        assertTrue(actual = thresholds.tunedLabels().isEmpty())
    }

    @Test
    internal fun `exposes its thresholds as a defensive copy`() {
        val thresholds = LabelThresholds(byLabel = mapOf(rare to 0.2), fallback = 0.5)

        assertEquals(expected = mapOf(rare to 0.2), actual = thresholds.asMap())
    }

    @Test
    internal fun `rejects a threshold outside zero to one`() {
        assertFailsWith<IllegalArgumentException> { LabelThresholds(byLabel = emptyMap(), fallback = 1.5) }
        assertFailsWith<IllegalArgumentException> { LabelThresholds(byLabel = mapOf(rare to -0.1), fallback = 0.5) }
    }
}
