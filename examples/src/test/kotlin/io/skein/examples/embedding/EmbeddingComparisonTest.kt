package io.skein.examples.embedding

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.spi.BatchVectorizer
import io.skein.classify.spi.Vectorizer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Exercises the comparison the embedding examples print, with a deterministic stand-in for a real
 * encoder — so the example's own wiring is verified without a model or a server.
 */
internal class EmbeddingComparisonTest {

    /**
     * A crude but real bag-of-words embedding: each vocabulary word owns a dimension. Enough
     * structure that a linear model can actually learn from it, and fully deterministic.
     */
    private class BagOfWordsVectorizer : Vectorizer {

        private val vocabulary = HashMap<String, Int>()
        private val width = 256

        override fun vectorize(text: String): FeatureVector {
            val counts = HashMap<Int, Float>()
            text.lowercase().split(" ").filter { token -> token.isNotBlank() }.forEach { token ->
                val index = vocabulary.getOrPut(key = token) { vocabulary.size % width }
                counts[index] = (counts[index] ?: 0.0f) + 1.0f
            }
            val dense = FloatArray(size = width)
            counts.forEach { (index, value) -> dense[index] = value }
            return FeatureVector(indices = IntArray(size = width) { it }, values = dense)
        }

        override fun dimension(): Int {
            return width
        }

        override fun fingerprint(): VectorizerFingerprint {
            return VectorizerFingerprint(kind = "bag-of-words", dimension = width, configDigest = "test")
        }
    }

    @Test
    internal fun `evaluates a vectorizer against the hand-written held-out set`() {
        val metrics = EmbeddingComparison.evaluate(vectorizer = BagOfWordsVectorizer())

        assertEquals(expected = 40, actual = metrics.sampleCount)
        assertTrue(actual = metrics.micro.precision in 0.0..1.0)
        assertTrue(actual = metrics.micro.recall in 0.0..1.0)
        assertTrue(actual = metrics.perLabel.isNotEmpty())
    }

    /** The same vectorizer, but announcing the batch capability through the port. */
    private class BatchingBagOfWordsVectorizer : BatchVectorizer {

        private val delegate = BagOfWordsVectorizer()

        var batchCalls: Int = 0
            private set

        override fun vectorize(text: String): FeatureVector = delegate.vectorize(text = text)

        override fun vectorizeAll(texts: List<String>): List<FeatureVector> {
            batchCalls += 1
            return texts.map { text -> delegate.vectorize(text = text) }
        }

        override fun dimension(): Int = delegate.dimension()

        override fun fingerprint(): VectorizerFingerprint = delegate.fingerprint()
    }

    @Test
    internal fun `batches through the port when the vectorizer supports it`() {
        val vectorizer = BatchingBagOfWordsVectorizer()

        val metrics = EmbeddingComparison.evaluate(vectorizer = vectorizer)

        // Once for the training corpus, once for the held-out set -- not once per record, which
        // for a real embedding service would be a round trip apiece.
        assertEquals(expected = 2, actual = vectorizer.batchCalls)
        assertEquals(expected = 40, actual = metrics.sampleCount)
    }

    @Test
    internal fun `the hashing baseline scores the same set`() {
        val hashing = EmbeddingComparison.hashingBaseline()

        assertEquals(expected = 40, actual = hashing.sampleCount)
        assertTrue(
            actual = hashing.micro.precision > 0.5,
            message = "the hashed baseline should still be usable, got ${hashing.micro.precision}",
        )
    }

    @Test
    internal fun `ranked recall grows with depth`() {
        val recalls = EmbeddingComparison.rankedRecall(vectorizer = BagOfWordsVectorizer())

        assertEquals(expected = listOf(1, 2, 3, 5), actual = recalls.keys.toList())
        assertTrue(
            actual = recalls.values.zipWithNext().all { (shallow, deep) -> deep >= shallow },
            message = "recall@k must be non-decreasing: $recalls",
        )
    }
}
