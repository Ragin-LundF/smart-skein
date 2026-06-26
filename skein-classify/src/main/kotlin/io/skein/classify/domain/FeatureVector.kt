package io.skein.classify.domain

/**
 * Sparse feature vector held as parallel primitive arrays — [indices] sorted ascending and
 * unique, [values] aligned by position. Primitive arrays avoid boxing and pointer chasing on
 * the classification hot path (see the performance strategy).
 */
class FeatureVector(val indices: IntArray, val values: FloatArray) {

    init {
        require(value = indices.size == values.size) { "indices and values must have equal length" }
    }

    /** Number of non-zero features. */
    fun nonZeroCount(): Int {
        return indices.size
    }
}
