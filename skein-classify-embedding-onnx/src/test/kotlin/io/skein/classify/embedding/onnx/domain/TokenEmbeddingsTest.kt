package io.skein.classify.embedding.onnx.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

internal class TokenEmbeddingsTest {

    private fun embeddings(vararg rows: Pair<FloatArray, Long>): TokenEmbeddings {
        return TokenEmbeddings(
            vectors = rows.map { (vector, _) -> vector }.toTypedArray(),
            attentionMask = rows.map { (_, flag) -> flag }.toLongArray(),
        )
    }

    /**
     * The measurement that matters: padding included in the mean would make a short text's vector
     * depend on how long the *other* texts in its batch happened to be.
     */
    @Test
    internal fun `mean pooling averages only the real tokens`() {
        val pooled = embeddings(
            floatArrayOf(1.0f, 3.0f) to 1L,
            floatArrayOf(3.0f, 5.0f) to 1L,
            floatArrayOf(99.0f, 99.0f) to 0L,
        ).pool(strategy = PoolingStrategyEnum.MEAN)

        assertEquals(expected = listOf(2.0f, 4.0f), actual = pooled.toList())
    }

    @Test
    internal fun `mean pooling of a single real token is that token`() {
        val pooled = embeddings(
            floatArrayOf(7.0f, -2.0f) to 1L,
            floatArrayOf(0.0f, 0.0f) to 0L,
        ).pool(strategy = PoolingStrategyEnum.MEAN)

        assertEquals(expected = listOf(7.0f, -2.0f), actual = pooled.toList())
    }

    @Test
    internal fun `CLS pooling takes the first position regardless of the mask`() {
        val pooled = embeddings(
            floatArrayOf(1.0f, 1.0f) to 1L,
            floatArrayOf(9.0f, 9.0f) to 1L,
        ).pool(strategy = PoolingStrategyEnum.CLS)

        assertEquals(expected = listOf(1.0f, 1.0f), actual = pooled.toList())
    }

    @Test
    internal fun `CLS pooling copies rather than aliasing the model's output`() {
        val first = floatArrayOf(1.0f, 1.0f)
        val pooled = embeddings(first to 1L).pool(strategy = PoolingStrategyEnum.CLS)

        pooled[0] = 42.0f

        assertEquals(expected = 1.0f, actual = first[0])
    }

    /** An empty or whitespace-only record masks to nothing; `NaN` here would poison every weight. */
    @Test
    internal fun `an all-padding sequence pools to zero rather than NaN`() {
        val pooled = embeddings(
            floatArrayOf(5.0f, 5.0f) to 0L,
            floatArrayOf(6.0f, 6.0f) to 0L,
        ).pool(strategy = PoolingStrategyEnum.MEAN)

        assertEquals(expected = listOf(0.0f, 0.0f), actual = pooled.toList())
    }

    @Test
    internal fun `reports the hidden size of its token vectors`() {
        assertEquals(expected = 3, actual = embeddings(floatArrayOf(1.0f, 2.0f, 3.0f) to 1L).hiddenSize())
    }

    @Test
    internal fun `rejects a malformed sequence`() {
        assertFailsWith<IllegalArgumentException> {
            TokenEmbeddings(vectors = emptyArray(), attentionMask = longArrayOf())
        }
        assertFailsWith<IllegalArgumentException> {
            TokenEmbeddings(vectors = arrayOf(floatArrayOf(1.0f)), attentionMask = longArrayOf(1L, 1L))
        }
        assertFailsWith<IllegalArgumentException> {
            TokenEmbeddings(
                vectors = arrayOf(floatArrayOf(1.0f), floatArrayOf(1.0f, 2.0f)),
                attentionMask = longArrayOf(1L, 1L),
            )
        }
    }
}
