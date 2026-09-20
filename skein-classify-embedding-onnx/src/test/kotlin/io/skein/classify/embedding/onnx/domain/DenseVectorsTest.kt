package io.skein.classify.embedding.onnx.domain

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class DenseVectorsTest {

    @Test
    internal fun `scales a vector to unit length`() {
        val normalized = DenseVectors.l2Normalize(vector = floatArrayOf(3.0f, 4.0f))

        assertEquals(expected = 0.6f, actual = normalized[0], absoluteTolerance = 1e-6f)
        assertEquals(expected = 0.8f, actual = normalized[1], absoluteTolerance = 1e-6f)
    }

    @Test
    internal fun `leaves direction unchanged while changing magnitude`() {
        val normalized = DenseVectors.l2Normalize(vector = floatArrayOf(2.0f, 0.0f, -2.0f))

        val length = sqrt(x = normalized.sumOf { c -> c.toDouble() * c })
        assertEquals(expected = 1.0, actual = length, absoluteTolerance = 1e-6)
        assertTrue(actual = normalized[0] > 0.0f && normalized[2] < 0.0f)
        assertEquals(expected = 0.0f, actual = normalized[1])
    }

    /** An empty record pools to zero; dividing by its norm would make every component NaN. */
    @Test
    internal fun `leaves the zero vector alone rather than producing NaN`() {
        val normalized = DenseVectors.l2Normalize(vector = floatArrayOf(0.0f, 0.0f, 0.0f))

        assertTrue(actual = normalized.all { component -> component == 0.0f })
    }

    @Test
    internal fun `does not mutate its input`() {
        val original = floatArrayOf(3.0f, 4.0f)

        DenseVectors.l2Normalize(vector = original)

        assertEquals(expected = listOf(3.0f, 4.0f), actual = original.toList())
    }

    @Test
    internal fun `maps component i to feature index i`() {
        val features = DenseVectors.asFeatureVector(vector = floatArrayOf(0.5f, -1.0f, 0.0f))

        assertEquals(expected = listOf(0, 1, 2), actual = features.indices.toList())
        assertEquals(expected = listOf(0.5f, -1.0f, 0.0f), actual = features.values.toList())
        assertEquals(expected = 3, actual = features.nonZeroCount())
    }
}
