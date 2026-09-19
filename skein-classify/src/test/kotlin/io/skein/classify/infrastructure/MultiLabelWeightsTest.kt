package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

internal class MultiLabelWeightsTest {

    private val labels = listOf(Label(value = "A"), Label(value = "B"))

    private fun weights(keepFraction: Double, vararg vectors: FloatArray): MultiLabelWeights {
        return MultiLabelWeights.fromLabelMajor(
            labels = labels,
            intercepts = doubleArrayOf(0.0, 0.0),
            weightsByLabel = vectors.toList(),
            featureCount = vectors.first().size,
            keepFraction = keepFraction,
        )
    }

    @Test
    internal fun `keeps every non-zero weight when nothing is pruned`() {
        val packed = weights(1.0, floatArrayOf(1.0f, 0.0f, 3.0f), floatArrayOf(4.0f, 5.0f, 0.0f))

        assertEquals(expected = 4, actual = packed.nonZeroCount())
        assertEquals(expected = 1.0, actual = packed.weightOf(feature = 0, labelIndex = 0))
        assertEquals(expected = 5.0, actual = packed.weightOf(feature = 1, labelIndex = 1))
        assertEquals(expected = 0.0, actual = packed.weightOf(feature = 1, labelIndex = 0))
    }

    /**
     * Magnitudes 1..8 with half kept puts the cutoff at 5.0, so the four strongest survive and the
     * four weakest go. This pins both ends: the largest weight is never pruned, and the smallest
     * surviving weight is the cutoff itself rather than anything below it.
     */
    @Test
    internal fun `pruning keeps the strongest weights and raises the weakest survivor`() {
        val packed = weights(
            0.5,
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f),
            floatArrayOf(5.0f, 6.0f, 7.0f, 8.0f),
        )

        assertEquals(expected = 4, actual = packed.nonZeroCount())
        assertEquals(expected = 8.0, actual = packed.weightOf(feature = 3, labelIndex = 1))
        assertEquals(expected = 5.0, actual = packed.weightOf(feature = 0, labelIndex = 1))
        assertEquals(expected = 0.0, actual = packed.weightOf(feature = 3, labelIndex = 0))
        assertEquals(expected = 0.0, actual = packed.weightOf(feature = 0, labelIndex = 0))
    }

    @Test
    internal fun `pruning ranks by magnitude so a large negative weight survives`() {
        val packed = weights(
            0.25,
            floatArrayOf(-9.0f, 0.5f),
            floatArrayOf(0.25f, 0.1f),
        )

        assertEquals(expected = 1, actual = packed.nonZeroCount())
        assertEquals(expected = -9.0, actual = packed.weightOf(feature = 0, labelIndex = 0))
    }

    @Test
    internal fun `ties at the cutoff are all kept`() {
        val packed = MultiLabelWeights.fromLabelMajor(
            labels = listOf(Label(value = "A")),
            intercepts = doubleArrayOf(0.0),
            weightsByLabel = listOf(floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)),
            featureCount = 4,
            keepFraction = 0.25,
        )

        // Asking for one of four identical weights keeps all four rather than picking arbitrarily.
        assertEquals(expected = 4, actual = packed.nonZeroCount())
    }

    @Test
    internal fun `intercepts survive pruning untouched`() {
        val packed = MultiLabelWeights.fromLabelMajor(
            labels = labels,
            intercepts = doubleArrayOf(-2.5, 7.25),
            weightsByLabel = listOf(floatArrayOf(0.001f), floatArrayOf(0.002f)),
            featureCount = 1,
            keepFraction = 0.01,
        )

        assertEquals(expected = -2.5, actual = packed.intercepts[0])
        assertEquals(expected = 7.25, actual = packed.intercepts[1])
    }

    @Test
    internal fun `scoring starts from the intercepts and accumulates only active features`() {
        val packed = MultiLabelWeights.fromLabelMajor(
            labels = labels,
            intercepts = doubleArrayOf(1.0, -1.0),
            weightsByLabel = listOf(floatArrayOf(2.0f, 0.0f), floatArrayOf(0.0f, 4.0f)),
            featureCount = 2,
            keepFraction = 1.0,
        )

        val scores = packed.logits(
            features = FeatureVector(indices = intArrayOf(0), values = floatArrayOf(3.0f)),
        )

        assertEquals(expected = 1.0 + 2.0 * 3.0, actual = scores[0], absoluteTolerance = 1e-12)
        assertEquals(expected = -1.0, actual = scores[1], absoluteTolerance = 1e-12)
    }

    @Test
    internal fun `a feature outside the model's space is skipped rather than rejected`() {
        val packed = weights(1.0, floatArrayOf(2.0f), floatArrayOf(3.0f))

        val scores = packed.logits(
            features = FeatureVector(indices = intArrayOf(0, 99), values = floatArrayOf(1.0f, 1.0f)),
        )

        assertEquals(expected = 2.0, actual = scores[0], absoluteTolerance = 1e-12)
        assertEquals(expected = 0.0, actual = packed.weightOf(feature = 99, labelIndex = 0))
        assertEquals(expected = 0.0, actual = packed.weightOf(feature = -1, labelIndex = 0))
    }

    @Test
    internal fun `rejects an inconsistent shape`() {
        assertFailsWith<IllegalArgumentException> {
            MultiLabelWeights.fromLabelMajor(
                labels = labels,
                intercepts = doubleArrayOf(0.0),
                weightsByLabel = listOf(floatArrayOf(1.0f), floatArrayOf(1.0f)),
                featureCount = 1,
                keepFraction = 1.0,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            MultiLabelWeights.fromLabelMajor(
                labels = labels,
                intercepts = doubleArrayOf(0.0, 0.0),
                weightsByLabel = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(1.0f)),
                featureCount = 2,
                keepFraction = 1.0,
            )
        }
        assertFailsWith<IllegalArgumentException> { weights(0.0, floatArrayOf(1.0f), floatArrayOf(1.0f)) }
        assertFailsWith<IllegalArgumentException> { weights(1.5, floatArrayOf(1.0f), floatArrayOf(1.0f)) }
    }

    @Test
    internal fun `rejects a model that knows no labels`() {
        assertFailsWith<IllegalArgumentException> {
            MultiLabelWeights(
                labels = emptyList(),
                intercepts = doubleArrayOf(),
                featureOffsets = intArrayOf(0),
                labelIndices = intArrayOf(),
                weights = floatArrayOf(),
            )
        }
    }
}
