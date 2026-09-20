package io.skein.classify.embedding.onnx.infrastructure

import io.skein.classify.embedding.onnx.application.EmbeddingCache
import io.skein.classify.embedding.onnx.domain.NormalizationEnum
import io.skein.classify.embedding.onnx.domain.PoolingStrategyEnum
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.deleteIfExists
import kotlin.io.path.writeBytes
import kotlin.math.sqrt
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The real thing: a genuine ONNX model through ONNX Runtime and a genuine Hugging Face tokenizer,
 * with no stand-ins anywhere.
 *
 * The fixtures are tiny but not toys. `tiny-embedding-model.onnx` is a token-embedding lookup —
 * which *is* a static embedding model, the Model2Vec-class approach that sits between n-grams and a
 * transformer — and `tiny-tokenizer.json` is an ordinary word-level Hugging Face tokenizer. A
 * transformer differs in what fills the embedding table, not in the interface this adapter talks
 * to, so everything exercised here is exercised the same way by a real encoder.
 *
 * The embedding table is `row r = [r, r + 0.5, -r, 1.0]`, which makes every expected value below
 * hand-computable. The tokenizer's vocabulary is `alpha=1, beta=2, gamma=3, delta=4, epsilon=5`.
 */
internal class OnnxEmbeddingPipelineIntegrationTest {

    private val modelPath: Path = copyResource(name = "tiny-embedding-model.onnx", suffix = ".onnx")
    private val tokenizerPath: Path = copyResource(name = "tiny-tokenizer.json", suffix = ".json")

    private fun copyResource(name: String, suffix: String): Path {
        val bytes = OnnxEmbeddingPipelineIntegrationTest::class.java
            .getResourceAsStream("/reference/$name")!!
            .readBytes()
        val path = Files.createTempFile("skein-onnx", suffix)
        path.writeBytes(bytes)
        return path
    }

    @AfterTest
    internal fun cleanUp() {
        modelPath.deleteIfExists()
        tokenizerPath.deleteIfExists()
    }

    private fun open(
        pooling: PoolingStrategyEnum = PoolingStrategyEnum.MEAN,
        normalization: NormalizationEnum = NormalizationEnum.NONE,
        cache: EmbeddingCache? = null,
    ): OnnxEmbeddingVectorizer {
        return OnnxEmbeddingVectorizer.open(
            modelPath = modelPath,
            tokenizerPath = tokenizerPath,
            pooling = pooling,
            normalization = normalization,
            cache = cache,
        )
    }

    @Test
    internal fun `reads the hidden size from the model rather than being told`() {
        open().use { vectorizer ->
            assertEquals(expected = 4, actual = vectorizer.dimension())
            assertEquals(expected = 4, actual = vectorizer.fingerprint().dimension)
        }
    }

    @Test
    internal fun `embeds text end to end through the real model`() {
        open().use { vectorizer ->
            // "alpha beta" -> ids [1, 2] -> rows [1,1.5,-1,1] and [2,2.5,-2,1] -> mean below.
            val features = vectorizer.vectorize(text = "alpha beta")

            assertEquals(expected = listOf(0, 1, 2, 3), actual = features.indices.toList())
            assertEquals(expected = listOf(1.5f, 2.0f, -1.5f, 1.0f), actual = features.values.toList())
        }
    }

    /**
     * The batch pads "gamma" to two positions. Row 0 of the embedding table is `[0, 0.5, 0, 1]`, so
     * averaging *including* the padding would give exactly `[1.5, 2.0, -1.5, 1.0]` — the same
     * vector as "alpha beta". Two unrelated texts colliding is precisely the bug an ignored
     * attention mask causes, and this asserts it does not happen.
     */
    @Test
    internal fun `honours the attention mask so a padded text does not collide with another`() {
        open().use { vectorizer ->
            val results = vectorizer.vectorizeAll(texts = listOf("alpha beta", "gamma"))

            assertEquals(expected = listOf(1.5f, 2.0f, -1.5f, 1.0f), actual = results[0].values.toList())
            assertEquals(expected = listOf(3.0f, 3.5f, -3.0f, 1.0f), actual = results[1].values.toList())
            assertTrue(actual = results[0].values.toList() != results[1].values.toList())
        }
    }

    @Test
    internal fun `a text embeds the same alone as it does inside a batch`() {
        open().use { vectorizer ->
            val alone = vectorizer.vectorize(text = "gamma")
            val batched = vectorizer.vectorizeAll(texts = listOf("alpha beta gamma delta", "gamma"))[1]

            assertEquals(expected = alone.values.toList(), actual = batched.values.toList())
        }
    }

    @Test
    internal fun `CLS pooling takes the first token through the real model`() {
        open(pooling = PoolingStrategyEnum.CLS).use { vectorizer ->
            val features = vectorizer.vectorize(text = "alpha beta")

            assertEquals(expected = listOf(1.0f, 1.5f, -1.0f, 1.0f), actual = features.values.toList())
        }
    }

    @Test
    internal fun `L2 normalisation yields unit-length vectors`() {
        open(normalization = NormalizationEnum.L2).use { vectorizer ->
            val features = vectorizer.vectorize(text = "alpha beta")

            val length = sqrt(x = features.values.sumOf { component -> component.toDouble() * component })
            assertEquals(expected = 1.0, actual = length, absoluteTolerance = 1e-6)
        }
    }

    @Test
    internal fun `an unknown word falls back to the unknown token rather than failing`() {
        open().use { vectorizer ->
            // "zzzz" is outside the vocabulary and maps to [UNK] = 0 -> row [0, 0.5, -0, 1].
            // The third component reads back as +0.0: mean pooling accumulates in double and
            // divides, and -0.0 + 0.0 is +0.0, so the sign of a zero does not survive. Harmless
            // arithmetically, but boxed Float equality distinguishes them, so state it.
            val features = vectorizer.vectorize(text = "zzzz")

            assertEquals(expected = listOf(0.0f, 0.5f, 0.0f, 1.0f), actual = features.values.toList())
        }
    }

    @Test
    internal fun `the cache returns the same vector the model produced`() {
        val cache = EmbeddingCache()
        open(cache = cache).use { vectorizer ->
            val first = vectorizer.vectorize(text = "delta epsilon")
            val second = vectorizer.vectorize(text = "delta epsilon")

            assertEquals(expected = first.values.toList(), actual = second.values.toList())
            assertEquals(expected = 1, actual = cache.size())
        }
    }

    /**
     * The failure the fingerprint exists to catch: a model file replaced in place under an unchanged
     * name. A filename- or version-based identity would miss it entirely.
     */
    @Test
    internal fun `a model swapped in place under the same name changes the fingerprint`() {
        val before = open().use { vectorizer -> vectorizer.fingerprint() }

        val tampered = Files.readAllBytes(modelPath).copyOf()
        // Flip a byte inside the embedding table's payload, leaving the file name and size alone.
        tampered[tampered.size - 1] = (tampered[tampered.size - 1] + 1).toByte()
        modelPath.writeBytes(tampered)

        val after = open().use { vectorizer -> vectorizer.fingerprint() }

        assertTrue(actual = before != after, message = "a tampered model kept its fingerprint")
    }

    @Test
    internal fun `the same files fingerprint identically across instances`() {
        val first = open().use { vectorizer -> vectorizer.fingerprint() }
        val second = open().use { vectorizer -> vectorizer.fingerprint() }

        assertEquals(expected = first, actual = second)
    }

    @Test
    internal fun `rejects a batch whose rows were not padded to one length`() {
        OnnxEmbeddingRuntime(modelPath = modelPath).use { runtime ->
            assertFailsWith<IllegalArgumentException> {
                runtime.encode(
                    batch = listOf(
                        io.skein.classify.embedding.onnx.domain.TokenizedText(
                            ids = longArrayOf(1L, 2L),
                            attentionMask = longArrayOf(1L, 1L),
                        ),
                        io.skein.classify.embedding.onnx.domain.TokenizedText(
                            ids = longArrayOf(3L),
                            attentionMask = longArrayOf(1L),
                        ),
                    ),
                )
            }
            assertFailsWith<IllegalArgumentException> { runtime.encode(batch = emptyList()) }
        }
    }

    @Test
    internal fun `a missing model file fails rather than yielding a broken vectorizer`() {
        assertFailsWith<Exception> {
            OnnxEmbeddingVectorizer.open(
                modelPath = modelPath.resolveSibling("does-not-exist.onnx"),
                tokenizerPath = tokenizerPath,
            )
        }
    }
}
