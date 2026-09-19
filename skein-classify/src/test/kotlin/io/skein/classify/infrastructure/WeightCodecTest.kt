package io.skein.classify.infrastructure

import io.skein.classify.domain.WeightEncodingEnum
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class WeightCodecTest {

    private val featureOffsets = intArrayOf(0, 3, 3, 5)
    private val labelIndices = intArrayOf(2, 7, 40, 1, 9)

    @Test
    internal fun `delta coding round-trips exactly`() {
        val deltas = WeightCodec.encodeLabelIndices(featureOffsets = featureOffsets, labelIndices = labelIndices)

        assertEquals(
            expected = labelIndices.toList(),
            actual = WeightCodec.decodeLabelIndices(featureOffsets = featureOffsets, deltas = deltas).toList(),
        )
    }

    @Test
    internal fun `delta coding restarts at each feature row`() {
        val deltas = WeightCodec.encodeLabelIndices(featureOffsets = featureOffsets, labelIndices = labelIndices)

        // Row 0 holds 2, 7, 40 -> 2, +5, +33. Row 1 is empty. Row 2 holds 1, 9 -> 1, +8 (absolute
        // again, not a delta from the previous row's 40).
        assertEquals(expected = listOf(2, 5, 33, 1, 8), actual = deltas.toList())
    }

    @Test
    internal fun `delta coding keeps gaps far smaller than the indices`() {
        val labels = 237
        val offsets = intArrayOf(0, 20)
        val indices = IntArray(size = 20) { step -> step * (labels / 20) }

        val deltas = WeightCodec.encodeLabelIndices(featureOffsets = offsets, labelIndices = indices)

        // Every gap fits in one variable-length byte, where the raw indices would not.
        assertTrue(actual = deltas.drop(n = 1).all { delta -> delta < 128 })
    }

    @Test
    internal fun `handles an empty weight matrix`() {
        val empty = WeightCodec.encodeLabelIndices(featureOffsets = intArrayOf(0, 0), labelIndices = intArrayOf())

        assertEquals(expected = 0, actual = empty.size)
        assertEquals(
            expected = 0,
            actual = WeightCodec.decodeLabelIndices(featureOffsets = intArrayOf(0, 0), deltas = empty).size,
        )
    }

    @Test
    internal fun `chooses half precision for ordinary fitted weights`() {
        val weights = floatArrayOf(0.0f, 1.5f, -12.25f, 1000.0f)

        assertEquals(
            expected = WeightEncodingEnum.FLOAT16,
            actual = WeightCodec.narrowestEncoding(weights = weights),
        )
    }

    /** Saturating to infinity to save two bytes would corrupt the model silently. */
    @Test
    internal fun `falls back to full precision when a weight exceeds the half-precision range`() {
        assertEquals(
            expected = WeightEncodingEnum.FLOAT32,
            actual = WeightCodec.narrowestEncoding(weights = floatArrayOf(1.0f, 70_000.0f)),
        )
        assertEquals(
            expected = WeightEncodingEnum.FLOAT32,
            actual = WeightCodec.narrowestEncoding(weights = floatArrayOf(Float.POSITIVE_INFINITY)),
        )
        assertEquals(
            expected = WeightEncodingEnum.FLOAT32,
            actual = WeightCodec.narrowestEncoding(weights = floatArrayOf(Float.NaN)),
        )
    }

    @Test
    internal fun `half precision halves the payload and stays within its precision`() {
        val weights = floatArrayOf(0.5f, -1.25f, 3.125f, 0.001f)

        val packed = WeightCodec.encodeWeights(weights = weights, encoding = WeightEncodingEnum.FLOAT16)
        val restored = WeightCodec.decodeWeights(bytes = packed, encoding = WeightEncodingEnum.FLOAT16)

        assertEquals(expected = weights.size * 2, actual = packed.size)
        weights.indices.forEach { i ->
            assertTrue(
                actual = abs(x = weights[i] - restored[i]) <= 1e-3f,
                message = "weight ${weights[i]} restored as ${restored[i]}",
            )
        }
    }

    @Test
    internal fun `full precision round-trips exactly`() {
        val weights = floatArrayOf(0.1f, -70_000.0f, 3.14159f)

        val packed = WeightCodec.encodeWeights(weights = weights, encoding = WeightEncodingEnum.FLOAT32)

        assertEquals(expected = weights.size * 4, actual = packed.size)
        assertEquals(
            expected = weights.toList(),
            actual = WeightCodec.decodeWeights(bytes = packed, encoding = WeightEncodingEnum.FLOAT32).toList(),
        )
    }

    @Test
    internal fun `half precision preserves the ordering of magnitudes`() {
        val weights = floatArrayOf(0.001f, 0.01f, 0.1f, 1.0f, 10.0f)

        val restored = WeightCodec.decodeWeights(
            bytes = WeightCodec.encodeWeights(weights = weights, encoding = WeightEncodingEnum.FLOAT16),
            encoding = WeightEncodingEnum.FLOAT16,
        )

        assertEquals(expected = restored.toList().sorted(), actual = restored.toList())
    }

    @Test
    internal fun `rejects a payload that is not a whole number of values`() {
        assertFailsWith<IllegalArgumentException> {
            WeightCodec.decodeWeights(bytes = byteArrayOf(1, 2, 3), encoding = WeightEncodingEnum.FLOAT16)
        }
    }
}
