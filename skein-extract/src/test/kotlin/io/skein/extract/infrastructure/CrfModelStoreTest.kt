package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.Token
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.GZIPInputStream
import kotlin.io.path.deleteIfExists
import kotlin.io.path.inputStream
import kotlin.io.path.readBytes
import kotlin.io.path.writeBytes
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class CrfModelStoreTest {

    private val modelFile: Path = Files.createTempFile("skein-crf", ".skeincrf")
    private val tokenizer = TypedTokenizer()

    @AfterTest
    fun cleanup() {
        modelFile.deleteIfExists()
    }

    private fun tokens(text: String): List<Token> {
        return tokenizer.tokenize(text = text)
    }

    private val trainingText = listOf(
        "invoice 4711 paid" to listOf("KEY", "VALUE", "O"),
        "invoice 4712 paid" to listOf("KEY", "VALUE", "O"),
        "receipt 9001 sent" to listOf("KEY", "VALUE", "O"),
        "receipt 9002 sent" to listOf("KEY", "VALUE", "O"),
    )

    /** Deterministic training: fixed corpus, fixed order, fixed epoch count, no RNG anywhere. */
    private fun trainedLabeler(decayRate: Double = 0.0): CrfSequenceLabeler {
        val labeler = CrfSequenceLabeler(initialLearningRate = 0.1, decayRate = decayRate)
        repeat(times = 40) {
            trainingText.forEach { (text, tags) ->
                labeler.learn(tokens = tokens(text = text), tags = tags.map { tag -> Tag(value = tag) })
            }
        }
        return labeler
    }

    private val probes = listOf(
        "invoice 4711 paid",
        "invoice 5555 paid",
        "receipt 1234 sent",
        "invoice",
        "",
    )

    @Test
    internal fun `a loaded model labels identically to the saved one`() {
        val original = trainedLabeler()
        CrfModelStore.save(
            path = modelFile,
            snapshot = original.snapshot(),
            retention = FeatureRetentionEnum.ALL_FEATURES,
        )
        val restored = CrfSequenceLabeler.from(snapshot = CrfModelStore.load(path = modelFile))

        probes.forEach { probe ->
            assertEquals(
                expected = original.label(tokens = tokens(text = probe)),
                actual = restored.label(tokens = tokens(text = probe)),
                message = "diverged on '$probe'",
            )
        }
    }

    @Test
    internal fun `a loaded snapshot equals the saved snapshot`() {
        val original = trainedLabeler()
        CrfModelStore.save(
            path = modelFile,
            snapshot = original.snapshot(),
            retention = FeatureRetentionEnum.ALL_FEATURES,
        )

        assertEquals(
            expected = original.snapshot(),
            actual = CrfSequenceLabeler.from(snapshot = CrfModelStore.load(path = modelFile)).snapshot(),
        )
    }

    @Test
    internal fun `training resumes identically after a reload`() {
        // A non-zero decay makes the next step size depend on the persisted step counter, so a model
        // that lost its hyperparameters or its progress diverges on the very next learn call.
        val original = trainedLabeler(decayRate = 0.01)
        CrfModelStore.save(
            path = modelFile,
            snapshot = original.snapshot(),
            retention = FeatureRetentionEnum.ALL_FEATURES,
        )
        val restored = CrfSequenceLabeler.from(snapshot = CrfModelStore.load(path = modelFile))

        val extra = tokens(text = "invoice 8888 paid")
        val extraTags = listOf("KEY", "VALUE", "O").map { tag -> Tag(value = tag) }
        original.learn(tokens = extra, tags = extraTags)
        restored.learn(tokens = extra, tags = extraTags)

        assertEquals(expected = original.snapshot().step, actual = restored.snapshot().step)
        assertEquals(expected = original.snapshot(), actual = restored.snapshot())
    }

    @Test
    internal fun `tag order survives exactly`() {
        val labeler = CrfSequenceLabeler()
        labeler.learn(
            tokens = tokens(text = "zeta alpha mid"),
            tags = listOf("ZED", "ALPHA", "MID").map { tag -> Tag(value = tag) },
        )
        CrfModelStore.save(path = modelFile, snapshot = labeler.snapshot())

        assertEquals(
            expected = listOf(Tag(value = "ZED"), Tag(value = "ALPHA"), Tag(value = "MID")),
            actual = CrfModelStore.load(path = modelFile).tagOrder,
        )
    }

    @Test
    internal fun `saving the same snapshot twice produces an identical payload`() {
        val snapshot = trainedLabeler().snapshot()
        CrfModelStore.save(path = modelFile, snapshot = snapshot)
        val first = payloadOf(path = modelFile)
        CrfModelStore.save(path = modelFile, snapshot = snapshot)
        val second = payloadOf(path = modelFile)

        // The creation timestamp is the leading 8 bytes and is deliberately not reproducible;
        // everything after it is, because every collection is written in a sorted or semantic order.
        assertContentEquals(
            expected = first.copyOfRange(fromIndex = 8, toIndex = first.size),
            actual = second.copyOfRange(fromIndex = 8, toIndex = second.size),
        )
    }

    @Test
    internal fun `metadata reads back without inflating the weights`() {
        CrfModelStore.save(
            path = modelFile,
            snapshot = trainedLabeler().snapshot(),
            retention = FeatureRetentionEnum.STRUCTURAL_ONLY,
            minLexicalOccurrences = 5,
        )
        val metadata = CrfModelStore.metadata(path = modelFile)

        assertEquals(expected = 1, actual = metadata.formatMajor)
        assertEquals(expected = 0, actual = metadata.formatMinor)
        assertEquals(expected = FeatureRetentionEnum.STRUCTURAL_ONLY, actual = metadata.retention)
        assertEquals(expected = 5, actual = metadata.minLexicalOccurrences)
    }

    // ---- privacy ------------------------------------------------------------------------------

    private fun payloadOf(path: Path): ByteArray {
        return path.inputStream().use { file ->
            file.readNBytes(6)
            GZIPInputStream(file).use { gzip ->
                val buffer = ByteArrayOutputStream()
                gzip.copyTo(out = buffer)
                buffer.toByteArray()
            }
        }
    }

    private fun ByteArray.containsBytesOf(text: String): Boolean {
        val needle = text.toByteArray(charset = Charsets.UTF_8)
        return (0..size - needle.size).any { start ->
            needle.indices.all { offset -> this[start + offset] == needle[offset] }
        }
    }

    private fun labelerWithRareToken(): CrfSequenceLabeler {
        val labeler = trainedLabeler()
        // A single occurrence of a distinctive, PII-shaped token.
        labeler.learn(
            tokens = tokens(text = "invoice DE89370400440532013000 paid"),
            tags = listOf("KEY", "VALUE", "O").map { tag -> Tag(value = tag) },
        )
        return labeler
    }

    @Test
    internal fun `structural-only writes no raw token text at all`() {
        CrfModelStore.save(
            path = modelFile,
            snapshot = labelerWithRareToken().snapshot(),
            retention = FeatureRetentionEnum.STRUCTURAL_ONLY,
        )
        val payload = payloadOf(path = modelFile)

        assertTrue(actual = !payload.containsBytesOf(text = "de89370400440532013000"), message = "IBAN leaked")
        assertTrue(actual = !payload.containsBytesOf(text = "invoice"), message = "vocabulary leaked")
        assertTrue(actual = payload.containsBytesOf(text = "type="), message = "structure was dropped too")
    }

    @Test
    internal fun `frequent-only drops a token seen once but keeps a repeated one`() {
        CrfModelStore.save(
            path = modelFile,
            snapshot = labelerWithRareToken().snapshot(),
            retention = FeatureRetentionEnum.FREQUENT_ONLY,
        )
        val payload = payloadOf(path = modelFile)

        assertTrue(actual = !payload.containsBytesOf(text = "de89370400440532013000"), message = "rare token kept")
        assertTrue(actual = payload.containsBytesOf(text = "invoice"), message = "frequent token dropped")
    }

    @Test
    internal fun `all-features keeps even a once-seen token`() {
        CrfModelStore.save(
            path = modelFile,
            snapshot = labelerWithRareToken().snapshot(),
            retention = FeatureRetentionEnum.ALL_FEATURES,
        )

        assertTrue(actual = payloadOf(path = modelFile).containsBytesOf(text = "de89370400440532013000"))
    }

    // ---- format robustness --------------------------------------------------------------------

    @Test
    internal fun `rejects a foreign magic`() {
        modelFile.writeBytes(array = byteArrayOf(0x53, 0x4B, 0x45, 0x49, 0x01, 0x00))
        val failure = assertFailsWith<IllegalArgumentException> { CrfModelStore.load(path = modelFile) }
        assertTrue(actual = failure.message.orEmpty().contains(other = "SKCR"))
    }

    @Test
    internal fun `rejects a newer major version and names both`() {
        CrfModelStore.save(path = modelFile, snapshot = trainedLabeler().snapshot())
        val bytes = modelFile.readBytes()
        bytes[4] = 2
        modelFile.writeBytes(array = bytes)

        val failure = assertFailsWith<IllegalArgumentException> { CrfModelStore.load(path = modelFile) }
        assertTrue(actual = failure.message.orEmpty().contains(other = "major 2"))
    }

    @Test
    internal fun `accepts a newer minor version`() {
        CrfModelStore.save(
            path = modelFile,
            snapshot = trainedLabeler().snapshot(),
            retention = FeatureRetentionEnum.ALL_FEATURES,
        )
        val bytes = modelFile.readBytes()
        bytes[5] = 99
        modelFile.writeBytes(array = bytes)

        assertEquals(expected = 99, actual = CrfModelStore.metadata(path = modelFile).formatMinor)
        assertTrue(actual = CrfModelStore.load(path = modelFile).tagOrder.isNotEmpty())
    }

    @Test
    internal fun `rejects a truncated file`() {
        CrfModelStore.save(path = modelFile, snapshot = trainedLabeler().snapshot())
        val bytes = modelFile.readBytes()
        modelFile.writeBytes(array = bytes.copyOfRange(fromIndex = 0, toIndex = 12))

        val failure = assertFailsWith<IllegalArgumentException> { CrfModelStore.load(path = modelFile) }
        assertTrue(actual = failure.message.orEmpty().contains(other = "SKCR"))
    }

    @Test
    internal fun `rejects saving an untrained labeler or a bad threshold`() {
        assertFailsWith<IllegalArgumentException> {
            CrfModelStore.save(path = modelFile, snapshot = CrfSequenceLabeler().snapshot())
        }
        assertFailsWith<IllegalArgumentException> {
            CrfModelStore.save(
                path = modelFile,
                snapshot = trainedLabeler().snapshot(),
                minLexicalOccurrences = 0,
            )
        }
    }
}
