package io.skein.classify.embedding.onnx.domain

/**
 * What the model produced for one text: a vector per token position, plus the mask saying which of
 * those positions were real.
 *
 * [vectors] is `[sequenceLength][hiddenSize]`, exactly the `last_hidden_state` row a sentence
 * encoder emits. Pooling it into a single vector is the last step before the classifier sees
 * anything, and it is pure arithmetic — which is why it lives here rather than inside the ONNX
 * adapter, where it could only be tested with a model loaded.
 */
class TokenEmbeddings(val vectors: Array<FloatArray>, val attentionMask: LongArray) {

    init {
        require(value = vectors.isNotEmpty()) { "a sequence must hold at least one token position" }
        require(value = vectors.size == attentionMask.size) {
            "vectors (${vectors.size}) and attentionMask (${attentionMask.size}) must have equal length"
        }
        require(value = vectors.all { vector -> vector.size == vectors[0].size }) {
            "every token vector must have the same hidden size"
        }
    }

    /** Width of one token's vector. */
    fun hiddenSize(): Int {
        return vectors[0].size
    }

    /**
     * Collapses the sequence to one vector under [strategy].
     *
     * [PoolingStrategyEnum.MEAN] averages only the positions the mask marks real. Including padding
     * would pull every short text towards the padding token's embedding, by an amount that depends
     * on how long the *other* texts in its batch happened to be — so the same text would embed
     * differently depending on what it was batched with, which is a genuinely confusing bug to
     * chase.
     *
     * A sequence whose mask is entirely zero has nothing to average and pools to the zero vector
     * rather than to `NaN`; that is what an empty or whitespace-only input produces.
     */
    fun pool(strategy: PoolingStrategyEnum): FloatArray {
        return when (strategy) {
            PoolingStrategyEnum.MEAN -> meanOfRealTokens()
            PoolingStrategyEnum.CLS -> vectors[0].copyOf()
        }
    }

    private fun meanOfRealTokens(): FloatArray {
        val totals = DoubleArray(size = hiddenSize())
        var counted = 0
        vectors.indices.forEach { position ->
            if (attentionMask[position] == 1L) {
                counted++
                val vector = vectors[position]
                vector.indices.forEach { component -> totals[component] += vector[component] }
            }
        }
        if (counted == 0) {
            return FloatArray(size = hiddenSize())
        }
        return FloatArray(size = totals.size) { component -> (totals[component] / counted).toFloat() }
    }
}
