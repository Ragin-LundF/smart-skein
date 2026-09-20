package io.skein.classify.embedding.http.domain

import io.skein.classify.domain.VectorizerCanary
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class EmbeddingServiceConfigTest {

    private fun config(): EmbeddingServiceConfig {
        return EmbeddingServiceConfig(baseUrl = "http://localhost:1234/v1", model = "m", modelRevision = "r")
    }

    @Test
    internal fun `builds the embeddings endpoint from the base url`() {
        assertEquals(expected = "http://localhost:1234/v1/embeddings", actual = config().embeddingsUrl())
    }

    @Test
    internal fun `defaults leave every optional behaviour off`() {
        val subject = config()

        assertEquals(expected = "", actual = subject.inputPrefix)
        assertEquals(expected = null, actual = subject.dimension)
        assertTrue(actual = subject.headers.isEmpty())
        assertTrue(actual = subject.canaryProbes.isEmpty())
        assertEquals(expected = VectorizerCanary.DEFAULT_TOLERANCE, actual = subject.canaryTolerance)
        assertEquals(expected = EmbeddingServiceConfig.DEFAULT_BATCH_SIZE, actual = subject.batchSize)
        assertEquals(expected = EmbeddingServiceConfig.DEFAULT_TIMEOUT_SECONDS, actual = subject.timeoutSeconds)
    }

    @Test
    internal fun `rejects a configuration that cannot identify its model`() {
        assertFailsWith<IllegalArgumentException> { config().copy(baseUrl = " ") }
        assertFailsWith<IllegalArgumentException> { config().copy(baseUrl = "http://x/v1/") }
        assertFailsWith<IllegalArgumentException> { config().copy(model = " ") }
        assertFailsWith<IllegalArgumentException> { config().copy(modelRevision = " ") }
        assertFailsWith<IllegalArgumentException> { config().copy(dimension = 0) }
        assertFailsWith<IllegalArgumentException> { config().copy(batchSize = 0) }
        assertFailsWith<IllegalArgumentException> { config().copy(timeoutSeconds = 0) }
        assertFailsWith<IllegalArgumentException> { config().copy(headers = mapOf(" " to "v")) }
        assertFailsWith<IllegalArgumentException> { config().copy(canaryProbes = listOf("a", "a")) }
    }

    @Test
    internal fun `prints without headers when none are set`() {
        val printed = config().toString()

        assertTrue(actual = printed.contains(other = "headers={}"))
        assertTrue(actual = printed.contains(other = "model=m"))
    }

    @Test
    internal fun `redacts every header value, not just the first`() {
        val printed = config().copy(headers = mapOf("Authorization" to "Bearer a", "X-Proxy-Token" to "b")).toString()

        assertFalse(actual = printed.contains(other = "Bearer a"))
        assertTrue(actual = printed.contains(other = "Authorization=***"))
        assertTrue(actual = printed.contains(other = "X-Proxy-Token=***"))
    }

    /**
     * The documented default. Its E5 prefix is mandatory rather than decorative, and it ships with
     * a canary because route B has no other way to notice a changed model.
     */
    @Test
    internal fun `the LM Studio preset pins everything that identifies the model`() {
        val subject = EmbeddingServiceConfig.lmStudioMultilingualE5Small()

        assertEquals(expected = "http://localhost:1234/v1", actual = subject.baseUrl)
        assertEquals(expected = "passage: ", actual = subject.inputPrefix)
        assertEquals(expected = 384, actual = subject.dimension)
        assertEquals(expected = EmbeddingProbes.DEFAULT, actual = subject.canaryProbes)
    }

    /** A probe set confined to one language cannot see a revision that shifts another. */
    @Test
    internal fun `the default probes are distinct, short and not all one language`() {
        val probes = EmbeddingProbes.DEFAULT

        assertTrue(actual = probes.size >= 2)
        assertEquals(expected = probes.size, actual = probes.distinct().size)
        assertTrue(actual = probes.all { probe -> probe.length <= 256 })
        assertTrue(actual = probes.any { probe -> probe.contains(other = "ü") })
    }
}
