package io.skein.classify.embedding.onnx.infrastructure

import io.skein.classify.embedding.onnx.application.EmbeddingCache
import io.skein.classify.embedding.onnx.domain.NormalizationEnum
import io.skein.classify.embedding.onnx.domain.PoolingStrategyEnum
import io.skein.classify.embedding.onnx.domain.TokenEmbeddings
import io.skein.classify.embedding.onnx.domain.TokenizedText
import io.skein.classify.embedding.onnx.spi.EmbeddingRuntime
import io.skein.classify.embedding.onnx.spi.EmbeddingTokenizer
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The composition logic — batching, caching, pooling, normalisation, fingerprinting — against
 * stand-ins for the two native libraries. The libraries themselves are exercised for real in
 * [OnnxEmbeddingPipelineIntegrationTest]; splitting it this way is what lets the behaviour that
 * matters be pinned precisely without a model on disk.
 */
internal class OnnxEmbeddingVectorizerTest {

    /** Embeds each text as the length of its first token id, so results are traceable to inputs. */
    private class RecordingRuntime(private val hidden: Int = 2) : EmbeddingRuntime {

        val batches = ArrayList<List<String>>()
        var closed = false
        val seenSizes = ArrayList<Int>()

        override fun hiddenSize(): Int {
            return hidden
        }

        override fun encode(batch: List<TokenizedText>): List<TokenEmbeddings> {
            seenSizes.add(element = batch.size)
            return batch.map { row ->
                val value = row.ids[0].toFloat()
                TokenEmbeddings(
                    vectors = arrayOf(FloatArray(size = hidden) { component -> value + component }),
                    attentionMask = longArrayOf(1L),
                )
            }
        }

        override fun close() {
            closed = true
        }
    }

    /** Maps a text to a single token id equal to its length, so a fake embedding is predictable. */
    private class LengthTokenizer(private val name: String = "fake") : EmbeddingTokenizer {

        var closed = false
        val calls = ArrayList<List<String>>()

        override fun tokenize(texts: List<String>): List<TokenizedText> {
            calls.add(element = texts)
            return texts.map { text ->
                TokenizedText(ids = longArrayOf(text.length.toLong()), attentionMask = longArrayOf(1L))
            }
        }

        override fun identity(): String {
            return name
        }

        override fun close() {
            closed = true
        }
    }

    private fun vectorizer(
        runtime: EmbeddingRuntime = RecordingRuntime(),
        tokenizer: EmbeddingTokenizer = LengthTokenizer(),
        digest: String = "model-a",
        pooling: PoolingStrategyEnum = PoolingStrategyEnum.MEAN,
        normalization: NormalizationEnum = NormalizationEnum.NONE,
        cache: EmbeddingCache? = null,
    ): OnnxEmbeddingVectorizer {
        return OnnxEmbeddingVectorizer(
            runtime = runtime,
            tokenizer = tokenizer,
            modelDigest = digest,
            pooling = pooling,
            normalization = normalization,
            cache = cache,
        )
    }

    @Test
    internal fun `produces a feature vector of the model's hidden size`() {
        val features = vectorizer().vectorize(text = "abc")

        assertEquals(expected = listOf(0, 1), actual = features.indices.toList())
        assertEquals(expected = listOf(3.0f, 4.0f), actual = features.values.toList())
    }

    @Test
    internal fun `reports the model's hidden size as its dimension`() {
        assertEquals(expected = 2, actual = vectorizer().dimension())
    }

    /** The performance claim in the KDoc, made checkable: many texts must cost one forward pass. */
    @Test
    internal fun `embeds a whole batch in a single model call`() {
        val runtime = RecordingRuntime()

        val results = vectorizer(runtime = runtime).vectorizeAll(texts = listOf("a", "bb", "ccc", "dddd"))

        assertEquals(expected = 4, actual = results.size)
        assertEquals(expected = listOf(4), actual = runtime.seenSizes)
    }

    @Test
    internal fun `keeps results aligned with the order of the input`() {
        val results = vectorizer().vectorizeAll(texts = listOf("a", "bbb", "cc"))

        assertEquals(expected = listOf(1.0f, 3.0f, 2.0f), actual = results.map { it.values[0] })
    }

    @Test
    internal fun `returns nothing for an empty batch without calling the model`() {
        val runtime = RecordingRuntime()

        assertTrue(actual = vectorizer(runtime = runtime).vectorizeAll(texts = emptyList()).isEmpty())
        assertTrue(actual = runtime.seenSizes.isEmpty())
    }

    @Test
    internal fun `applies L2 normalisation when asked`() {
        val features = vectorizer(normalization = NormalizationEnum.L2).vectorize(text = "abc")

        val length = sqrt(x = features.values.sumOf { component -> component.toDouble() * component })
        assertEquals(expected = 1.0, actual = length, absoluteTolerance = 1e-6)
    }

    @Test
    internal fun `serves a repeated text from the cache instead of the model`() {
        val runtime = RecordingRuntime()
        val cache = EmbeddingCache()
        val subject = vectorizer(runtime = runtime, cache = cache)

        val first = subject.vectorize(text = "repeated")
        val second = subject.vectorize(text = "repeated")

        assertEquals(expected = first.values.toList(), actual = second.values.toList())
        assertEquals(expected = listOf(1), actual = runtime.seenSizes, message = "the model ran twice")
    }

    @Test
    internal fun `sends only the uncached texts of a mixed batch to the model`() {
        val runtime = RecordingRuntime()
        val cache = EmbeddingCache()
        val subject = vectorizer(runtime = runtime, cache = cache)
        subject.vectorizeAll(texts = listOf("aa", "bbb"))
        runtime.seenSizes.clear()

        val results = subject.vectorizeAll(texts = listOf("aa", "cccc", "bbb", "ddddd"))

        assertEquals(expected = listOf(2), actual = runtime.seenSizes)
        assertEquals(expected = listOf(2.0f, 4.0f, 3.0f, 5.0f), actual = results.map { it.values[0] })
    }

    /** A cache entry computed under one configuration must never be served under another. */
    @Test
    internal fun `does not serve a cached vector across two different configurations`() {
        val cache = EmbeddingCache()
        vectorizer(digest = "model-a", cache = cache).vectorize(text = "shared")
        val runtime = RecordingRuntime()

        vectorizer(runtime = runtime, digest = "model-b", cache = cache).vectorize(text = "shared")

        assertEquals(expected = listOf(1), actual = runtime.seenSizes, message = "a stale vector was reused")
        assertEquals(expected = 2, actual = cache.size())
    }

    @Test
    internal fun `every input to the vector changes the fingerprint`() {
        val baseline = vectorizer().fingerprint()

        assertTrue(actual = vectorizer(digest = "model-b").fingerprint() != baseline)
        assertTrue(actual = vectorizer(tokenizer = LengthTokenizer(name = "other")).fingerprint() != baseline)
        assertTrue(actual = vectorizer(pooling = PoolingStrategyEnum.CLS).fingerprint() != baseline)
        assertTrue(actual = vectorizer(normalization = NormalizationEnum.L2).fingerprint() != baseline)
        assertTrue(actual = vectorizer(runtime = RecordingRuntime(hidden = 8)).fingerprint() != baseline)
    }

    @Test
    internal fun `an identical configuration fingerprints identically`() {
        assertEquals(expected = vectorizer().fingerprint(), actual = vectorizer().fingerprint())
    }

    @Test
    internal fun `declares itself as an onnx embedding of the model's width`() {
        val fingerprint = vectorizer().fingerprint()

        assertEquals(expected = "onnx-embedding", actual = fingerprint.kind)
        assertEquals(expected = 2, actual = fingerprint.dimension)
    }

    @Test
    internal fun `closes both native resources`() {
        val runtime = RecordingRuntime()
        val tokenizer = LengthTokenizer()

        vectorizer(runtime = runtime, tokenizer = tokenizer).close()

        assertTrue(actual = runtime.closed)
        assertTrue(actual = tokenizer.closed)
    }
}
