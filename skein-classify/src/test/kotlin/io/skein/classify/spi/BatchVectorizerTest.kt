package io.skein.classify.spi

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.VectorizerFingerprint
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class BatchVectorizerTest {

    /** Plain vectorizer: no batching capability, so the extension must fall back to a loop. */
    private open class PlainVectorizer : Vectorizer {
        var singleCalls: Int = 0
            private set

        override fun vectorize(text: String): FeatureVector {
            singleCalls += 1
            return FeatureVector(indices = intArrayOf(0), values = floatArrayOf(text.length.toFloat()))
        }

        override fun dimension(): Int = 1

        override fun fingerprint(): VectorizerFingerprint =
            VectorizerFingerprint(kind = "plain", dimension = 1, configDigest = "d")
    }

    /** Batching vectorizer: one call for the whole list, whatever its size. */
    private class BatchingVectorizer : PlainVectorizer(), BatchVectorizer {
        var batchCalls: Int = 0
            private set

        override fun vectorizeAll(texts: List<String>): List<FeatureVector> {
            batchCalls += 1
            return texts.map { text ->
                FeatureVector(indices = intArrayOf(0), values = floatArrayOf(text.length.toFloat()))
            }
        }
    }

    @Test
    internal fun `a batching vectorizer is asked once for the whole corpus`() {
        val subject = BatchingVectorizer()

        val vectors: List<FeatureVector> = (subject as Vectorizer).vectorizeAll(texts = listOf("a", "bb", "ccc"))

        assertEquals(expected = 1, actual = subject.batchCalls)
        assertEquals(expected = 0, actual = subject.singleCalls)
        assertEquals(expected = listOf(1.0f, 2.0f, 3.0f), actual = vectors.map { vector -> vector.values[0] })
    }

    /**
     * The fallback is the point of the extension: library code featurising a corpus must not have
     * to know, or cast, to work with a vectorizer that cannot batch.
     */
    @Test
    internal fun `a plain vectorizer falls back to one call per text`() {
        val subject = PlainVectorizer()

        val vectors = subject.vectorizeAll(texts = listOf("a", "bb", "ccc"))

        assertEquals(expected = 3, actual = subject.singleCalls)
        assertEquals(expected = listOf(1.0f, 2.0f, 3.0f), actual = vectors.map { vector -> vector.values[0] })
    }

    @Test
    internal fun `an empty corpus contacts nothing on either path`() {
        val batching = BatchingVectorizer()
        val plain = PlainVectorizer()

        assertTrue(actual = (batching as Vectorizer).vectorizeAll(texts = emptyList()).isEmpty())
        assertTrue(actual = plain.vectorizeAll(texts = emptyList()).isEmpty())
        assertEquals(expected = 0, actual = plain.singleCalls)
    }

    /**
     * The member wins over the extension when the static type says `BatchVectorizer`, and both do
     * the same thing. That equivalence is what makes it safe to share the name.
     */
    @Test
    internal fun `the member and the extension agree`() {
        val subject = BatchingVectorizer()
        val texts = listOf("alpha", "beta")

        val viaMember = subject.vectorizeAll(texts = texts)
        val viaExtension = (subject as Vectorizer).vectorizeAll(texts = texts)

        assertEquals(
            expected = viaMember.map { vector -> vector.values.toList() },
            actual = viaExtension.map { vector -> vector.values.toList() },
        )
    }

    @Test
    internal fun `every batching vectorizer is usable wherever a Vectorizer is expected`() {
        val subject: Vectorizer = BatchingVectorizer()

        assertEquals(expected = 1, actual = subject.dimension())
        assertEquals(expected = "plain", actual = subject.fingerprint().kind)
    }
}
