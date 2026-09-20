package io.skein.classify.embedding.http.infrastructure

import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.spi.EmbeddingTransport
import io.skein.classify.embedding.http.spi.EmbeddingTransportResponse
import java.io.IOException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The same vectorizer driven through a fake [EmbeddingTransport] rather than a real server.
 *
 * This is what the port is for: the failure paths that matter — a body that is not JSON, a
 * transport that throws rather than answers — are a line of setup here and a fixture apiece
 * against an HTTP server.
 */
internal class FakeTransportVectorizerTest {

    private class FakeTransport(
        private val respond: (String) -> EmbeddingTransportResponse,
    ) : EmbeddingTransport {

        var lastUrl: String? = null
            private set

        var lastHeaders: Map<String, String> = emptyMap()
            private set

        override fun post(url: String, body: String, headers: Map<String, String>): EmbeddingTransportResponse {
            lastUrl = url
            lastHeaders = headers
            return respond(body)
        }
    }

    private fun config(dimension: Int? = 2): EmbeddingServiceConfig {
        return EmbeddingServiceConfig(
            baseUrl = "http://service/v1",
            model = "m",
            modelRevision = "r",
            dimension = dimension,
        )
    }

    private fun ok(body: String): EmbeddingTransportResponse {
        return EmbeddingTransportResponse(statusCode = 200, body = body)
    }

    @Test
    internal fun `posts to the configured endpoint with the configured headers`() {
        val transport = FakeTransport { ok(body = """{"data":[{"index":0,"embedding":[1.0,2.0]}]}""") }
        val subject = HttpEmbeddingVectorizer(
            config = config().copy(headers = mapOf("Authorization" to "Bearer t")),
            transport = transport,
        )

        subject.vectorize(text = "x")

        assertEquals(expected = "http://service/v1/embeddings", actual = transport.lastUrl)
        assertEquals(expected = mapOf("Authorization" to "Bearer t"), actual = transport.lastHeaders)
    }

    @Test
    internal fun `reports a body that is not JSON at all`() {
        val subject = HttpEmbeddingVectorizer(config = config(), transport = FakeTransport { ok(body = "not json") })

        val failure = assertFailsWith<IllegalStateException> { subject.vectorize(text = "x") }

        assertTrue(actual = failure.message!!.contains(other = "not an OpenAI embeddings response"))
    }

    @Test
    internal fun `reports JSON that is missing the data array`() {
        val subject = HttpEmbeddingVectorizer(
            config = config(),
            transport = FakeTransport { ok(body = """{"error":{"message":"model not loaded"}}""") },
        )

        assertFailsWith<IllegalStateException> { subject.vectorize(text = "x") }
    }

    /**
     * A transport failure must not surface as anything that looks like a model problem: "the
     * service is down" and "your model changed" call for completely different responses.
     */
    @Test
    internal fun `a transport failure propagates as itself`() {
        val subject = HttpEmbeddingVectorizer(
            config = config(),
            transport = FakeTransport { throw IOException("connection refused") },
        )

        assertFailsWith<IOException> { subject.vectorize(text = "x") }
        assertFalse(actual = subject.isReachable())
    }

    @Test
    internal fun `an unknown width is not checked, and is remembered once probed`() {
        var calls = 0
        val transport = FakeTransport {
            calls += 1
            ok(body = """{"data":[{"index":0,"embedding":[1.0,2.0,3.0]}]}""")
        }
        val subject = HttpEmbeddingVectorizer(config = config(dimension = null), transport = transport)

        assertEquals(expected = 3, actual = subject.dimension())
        assertEquals(expected = 3, actual = subject.dimension())
        assertEquals(expected = 1, actual = calls)
    }
}
