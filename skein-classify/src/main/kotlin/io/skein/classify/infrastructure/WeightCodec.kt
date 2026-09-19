package io.skein.classify.infrastructure

import io.skein.classify.domain.WeightEncodingEnum
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** Largest magnitude `float16` can represent; beyond it the encoding saturates to infinity. */
private const val FLOAT16_MAX_MAGNITUDE = 65504.0f

private const val FLOAT16_BYTES = 2
private const val FLOAT32_BYTES = 4

/**
 * Packs a fitted weight matrix for storage, and unpacks it again.
 *
 * Two independent savings, both measured on the reference model and both free of accuracy cost:
 *
 * - **Values as `float16`** halves the largest section of the file. See [WeightEncodingEnum.FLOAT16].
 * - **Label indices delta-coded within each feature row.** The indices in one row are ascending, so
 *   the gaps are far smaller than the values, and the small non-negative integers that result are
 *   exactly what the container's variable-length integer encoding is efficient at. A model with a
 *   few hundred labels stores most gaps in a single byte, which beats a fixed 16-bit index as well
 *   as the 32-bit one it replaces.
 *
 * Both are pure re-encodings: [decodeLabelIndices] reverses [encodeLabelIndices] exactly, and
 * `float16` round-trips to the nearest representable value with no reordering of magnitudes.
 */
object WeightCodec {

    /**
     * Rewrites [labelIndices] so that the first entry of each feature row is absolute and the rest
     * are gaps from the previous entry.
     *
     * [featureOffsets] delimits the rows, and the encoding is per row rather than across the whole
     * array so that decoding needs no state beyond the row it is in.
     */
    fun encodeLabelIndices(featureOffsets: IntArray, labelIndices: IntArray): IntArray {
        val encoded = IntArray(size = labelIndices.size)
        for (feature in 0 until featureOffsets.size - 1) {
            val from = featureOffsets[feature]
            val to = featureOffsets[feature + 1]
            var previous = 0
            for (k in from until to) {
                encoded[k] = if (k == from) labelIndices[k] else labelIndices[k] - previous
                previous = labelIndices[k]
            }
        }
        return encoded
    }

    /** Reverses [encodeLabelIndices]. */
    fun decodeLabelIndices(featureOffsets: IntArray, deltas: IntArray): IntArray {
        val decoded = IntArray(size = deltas.size)
        for (feature in 0 until featureOffsets.size - 1) {
            val from = featureOffsets[feature]
            val to = featureOffsets[feature + 1]
            var running = 0
            for (k in from until to) {
                running = if (k == from) deltas[k] else running + deltas[k]
                decoded[k] = running
            }
        }
        return decoded
    }

    /**
     * The narrowest encoding that represents every one of [weights] without saturating.
     *
     * Checked rather than assumed: a weight beyond `float16`'s range would encode to infinity, and
     * the resulting model would score every record carrying that feature as certain. Silently
     * corrupting a model to save bytes is not a trade worth making.
     */
    fun narrowestEncoding(weights: FloatArray): WeightEncodingEnum {
        val representable = weights.all { weight ->
            weight.isFinite() && abs(x = weight) <= FLOAT16_MAX_MAGNITUDE
        }
        return if (representable) WeightEncodingEnum.FLOAT16 else WeightEncodingEnum.FLOAT32
    }

    fun encodeWeights(weights: FloatArray, encoding: WeightEncodingEnum): ByteArray {
        val width = if (encoding == WeightEncodingEnum.FLOAT16) FLOAT16_BYTES else FLOAT32_BYTES
        val buffer = ByteBuffer.allocate(weights.size * width).order(ByteOrder.LITTLE_ENDIAN)
        weights.forEach { weight ->
            when (encoding) {
                WeightEncodingEnum.FLOAT16 -> buffer.putShort(java.lang.Float.floatToFloat16(weight))
                WeightEncodingEnum.FLOAT32 -> buffer.putFloat(weight)
            }
        }
        return buffer.array()
    }

    fun decodeWeights(bytes: ByteArray, encoding: WeightEncodingEnum): FloatArray {
        val width = if (encoding == WeightEncodingEnum.FLOAT16) FLOAT16_BYTES else FLOAT32_BYTES
        require(value = bytes.size % width == 0) {
            "weight payload of ${bytes.size} bytes is not a whole number of $encoding values"
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(size = bytes.size / width) {
            when (encoding) {
                WeightEncodingEnum.FLOAT16 -> java.lang.Float.float16ToFloat(buffer.getShort())
                WeightEncodingEnum.FLOAT32 -> buffer.getFloat()
            }
        }
    }
}
