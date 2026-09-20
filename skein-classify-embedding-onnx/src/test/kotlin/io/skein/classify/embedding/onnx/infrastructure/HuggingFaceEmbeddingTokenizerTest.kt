package io.skein.classify.embedding.onnx.infrastructure

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.deleteIfExists
import kotlin.io.path.writeBytes
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The real Hugging Face tokenizer against a real `tokenizer.json`: a word-level vocabulary of
 * `alpha=1, beta=2, gamma=3, delta=4, epsilon=5`, with `[UNK]=0` and a lowercasing normalizer.
 */
internal class HuggingFaceEmbeddingTokenizerTest {

    private val tokenizerPath: Path = Files.createTempFile("skein-tokenizer", ".json").also { path ->
        path.writeBytes(
            HuggingFaceEmbeddingTokenizerTest::class.java
                .getResourceAsStream("/reference/tiny-tokenizer.json")!!
                .readBytes(),
        )
    }

    @AfterTest
    internal fun cleanUp() {
        tokenizerPath.deleteIfExists()
    }

    private fun tokenizer(maxTokens: Int = 512): HuggingFaceEmbeddingTokenizer {
        return HuggingFaceEmbeddingTokenizer(tokenizerPath = tokenizerPath, maxTokens = maxTokens)
    }

    @Test
    internal fun `maps known words to their vocabulary ids`() {
        tokenizer().use { subject ->
            val row = subject.tokenize(texts = listOf("alpha beta gamma")).single()

            assertEquals(expected = listOf(1L, 2L, 3L), actual = row.ids.toList())
            assertEquals(expected = listOf(1L, 1L, 1L), actual = row.attentionMask.toList())
        }
    }

    @Test
    internal fun `maps an out-of-vocabulary word to the unknown token`() {
        tokenizer().use { subject ->
            val row = subject.tokenize(texts = listOf("alpha zzzz")).single()

            assertEquals(expected = listOf(1L, 0L), actual = row.ids.toList())
            assertEquals(expected = 2, actual = row.realTokenCount())
        }
    }

    /** A batch must be rectangular whether or not the model's own config enables padding. */
    @Test
    internal fun `pads a batch to its longest row and marks the padding`() {
        tokenizer().use { subject ->
            val rows = subject.tokenize(texts = listOf("alpha beta gamma", "delta"))

            assertTrue(actual = rows.all { row -> row.ids.size == 3 })
            assertEquals(expected = listOf(4L, 0L, 0L), actual = rows[1].ids.toList())
            assertEquals(expected = listOf(1L, 0L, 0L), actual = rows[1].attentionMask.toList())
            assertEquals(expected = 1, actual = rows[1].realTokenCount())
        }
    }

    @Test
    internal fun `applies the normalizer declared in the tokenizer file`() {
        tokenizer().use { subject ->
            val upper = subject.tokenize(texts = listOf("ALPHA BETA")).single()
            val lower = subject.tokenize(texts = listOf("alpha beta")).single()

            assertEquals(expected = lower.ids.toList(), actual = upper.ids.toList())
        }
    }

    /** Attention is quadratic in sequence length, so one outlier must not dominate its batch. */
    @Test
    internal fun `truncates a text beyond the token limit`() {
        tokenizer(maxTokens = 2).use { subject ->
            val row = subject.tokenize(texts = listOf("alpha beta gamma delta epsilon")).single()

            assertEquals(expected = listOf(1L, 2L), actual = row.ids.toList())
            assertEquals(expected = 2, actual = row.realTokenCount())
        }
    }

    @Test
    internal fun `identity covers the vocabulary and the token limit`() {
        val baseline = tokenizer().use { subject -> subject.identity() }
        val narrower = tokenizer(maxTokens = 8).use { subject -> subject.identity() }

        assertEquals(expected = baseline, actual = tokenizer().use { subject -> subject.identity() })
        assertTrue(actual = baseline != narrower)
    }

    @Test
    internal fun `a changed vocabulary changes the identity`() {
        val before = tokenizer().use { subject -> subject.identity() }
        tokenizerPath.writeBytes(
            Files.readAllBytes(tokenizerPath).decodeToString()
                .replace(oldValue = "\"epsilon\": 5", newValue = "\"zeta\": 5")
                .toByteArray(),
        )

        val after = tokenizer().use { subject -> subject.identity() }

        assertTrue(actual = before != after, message = "a different vocabulary kept its identity")
    }

    @Test
    internal fun `rejects an empty batch or a non-positive token limit`() {
        tokenizer().use { subject ->
            assertFailsWith<IllegalArgumentException> { subject.tokenize(texts = emptyList()) }
        }
        assertFailsWith<IllegalArgumentException> { tokenizer(maxTokens = 0) }
    }
}
