package io.skein.classify.embedding.onnx.domain

import io.skein.classify.domain.FeatureVector
import kotlin.math.sqrt

/**
 * The bridge from a dense embedding to the sparse representation the rest of the library speaks.
 */
object DenseVectors {

    /**
     * Scales [vector] to unit L2 length, or returns it unchanged when it is the zero vector.
     *
     * The zero case is real rather than defensive: an empty or whitespace-only record pools to
     * zero, and dividing by its norm would turn every component into `NaN` — which then propagates
     * silently through training into a model whose weights are all `NaN`.
     */
    fun l2Normalize(vector: FloatArray): FloatArray {
        var sumOfSquares = 0.0
        vector.forEach { component -> sumOfSquares += component.toDouble() * component }
        if (sumOfSquares == 0.0) {
            return vector.copyOf()
        }
        val norm = sqrt(x = sumOfSquares)
        return FloatArray(size = vector.size) { index -> (vector[index] / norm).toFloat() }
    }

    /**
     * Expresses a dense embedding as a [FeatureVector], component `i` becoming feature index `i`.
     *
     * **Nothing downstream changes.** `FeatureVector` is sparse, so a dense vector stored this way
     * spends one index per component — a real but small cost, and in exchange the scoring loop,
     * `LogisticObjective` and L-BFGS all work on an embedding exactly as they work on hashed
     * n-grams. No branch, no second code path, no dense variant of the learner.
     *
     * Zero components are kept rather than dropped. They cost a multiply that contributes nothing,
     * but dropping them would make a record's feature count depend on its values, and an embedding
     * has almost no exact zeros anyway.
     */
    fun asFeatureVector(vector: FloatArray): FeatureVector {
        return FeatureVector(
            indices = IntArray(size = vector.size) { index -> index },
            values = vector.copyOf(),
        )
    }
}
