package io.skein.examples.embedding

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Runs against a real HTTP server from the JDK, speaking the OpenAI embeddings schema, so the wire
 * format, batching, ordering and error handling are exercised without LM Studio or any network.
 */
internal class HttpEmbeddingVectorizerTest {

    private lateinit var server: HttpServer
    private lateinit var baseUrl: String

    /** Inputs of every request the server received, in order. */
    private val received = mutableListOf<List<String>>()
    private var status = 200
    private var body: ((List<String>) -> String)? = null

    @BeforeTest
    internal fun startServer() {
        received.clear()
        status = 200
        body = null
        server = HttpServer.create(InetSocketAddress(0), 0)
        server.createContext("/v1/embeddings") { exchange -> handle(exchange = exchange) }
        server.start()
        baseUrl = "http://localhost:${server.address.port}/v1"
    }

    @AfterTest
    internal fun stopServer() {
        server.stop(0)
    }

    private fun handle(exchange: HttpExchange) {
        val request = exchange.requestBody.readBytes().decodeToString()
        val inputs = Regex(pattern = "\"input\":\\[(.*?)]").find(request)?.groupValues?.get(1)
            ?.split(",")
            ?.map { raw -> raw.trim().trim('"') }
            ?: emptyList()
        received.add(element = inputs)

        val payload = body?.invoke(inputs) ?: defaultBody(inputs = inputs)
        val bytes = payload.toByteArray()
        exchange.sendResponseHeaders(status, bytes.size.toLong())
        exchange.responseBody.use { stream -> stream.write(bytes) }
    }

    /** Embeds each input as `[length, index, 0.5]`, so a vector is traceable to its request. */
    private fun defaultBody(inputs: List<String>): String {
        val entries = inputs.mapIndexed { index, text ->
            """{"index":$index,"embedding":[${text.length}.0,$index.0,0.5]}"""
        }
        return """{"object":"list","model":"test","data":[${entries.joinToString(separator = ",")}]}"""
    }

    private fun config(
        dimension: Int? = 3,
        batchSize: Int = 32,
        prefix: String = "",
        model: String = "test-model",
        revision: String = "rev-1",
    ): EmbeddingServiceConfig {
        return EmbeddingServiceConfig(
            baseUrl = baseUrl,
            model = model,
            modelRevision = revision,
            inputPrefix = prefix,
            dimension = dimension,
            batchSize = batchSize,
        )
    }

    private fun vectorizer(config: EmbeddingServiceConfig = config()): HttpEmbeddingVectorizer {
        return HttpEmbeddingVectorizer(config = config)
    }

    @Test
    internal fun `embeds a text through the service`() {
        val features = vectorizer().vectorize(text = "abcd")

        assertEquals(expected = listOf(0, 1, 2), actual = features.indices.toList())
        assertEquals(expected = listOf(4.0f, 0.0f, 0.5f), actual = features.values.toList())
    }

    @Test
    internal fun `sends one request per batch rather than one per text`() {
        vectorizer(config = config(batchSize = 2)).vectorizeAll(texts = listOf("a", "bb", "ccc", "dddd", "e"))

        assertEquals(expected = 3, actual = received.size)
        assertEquals(expected = listOf(2, 2, 1), actual = received.map { batch -> batch.size })
    }

    @Test
    internal fun `keeps results aligned with the input order across batches`() {
        val results = vectorizer(config = config(batchSize = 2))
            .vectorizeAll(texts = listOf("a", "bb", "ccc", "dddd"))

        assertEquals(expected = listOf(1.0f, 2.0f, 3.0f, 4.0f), actual = results.map { it.values[0] })
    }

    /**
     * The OpenAI schema carries an index because a server may answer out of order. Trusting arrival
     * order would attach every vector to the wrong record, silently.
     */
    @Test
    internal fun `reorders a response that comes back shuffled`() {
        body = { inputs ->
            val entries = inputs.indices.reversed().map { index ->
                """{"index":$index,"embedding":[${inputs[index].length}.0,$index.0,0.5]}"""
            }
            """{"data":[${entries.joinToString(separator = ",")}]}"""
        }

        val results = vectorizer().vectorizeAll(texts = listOf("a", "bb", "ccc"))

        assertEquals(expected = listOf(1.0f, 2.0f, 3.0f), actual = results.map { it.values[0] })
    }

    @Test
    internal fun `applies the model's required input prefix`() {
        vectorizer(config = config(prefix = "passage: ")).vectorize(text = "lasagne")

        assertEquals(expected = listOf("passage: lasagne"), actual = received.single())
    }

    @Test
    internal fun `sends nothing for an empty batch`() {
        assertTrue(actual = vectorizer().vectorizeAll(texts = emptyList()).isEmpty())
        assertTrue(actual = received.isEmpty())
    }

    @Test
    internal fun `learns the dimension from the service when it is not configured`() {
        assertEquals(expected = 3, actual = vectorizer(config = config(dimension = null)).dimension())
    }

    @Test
    internal fun `uses the configured dimension without probing`() {
        assertEquals(expected = 3, actual = vectorizer().dimension())
        assertTrue(actual = received.isEmpty())
    }

    @Test
    internal fun `fails loudly on an error status instead of returning a broken vector`() {
        status = 500
        body = { "upstream exploded" }

        val failure = assertFailsWith<IllegalStateException> { vectorizer().vectorize(text = "x") }

        assertTrue(actual = failure.message!!.contains(other = "HTTP 500"))
    }

    @Test
    internal fun `fails when the service returns the wrong number of embeddings`() {
        body = { """{"data":[{"index":0,"embedding":[1.0,2.0,3.0]}]}""" }

        assertFailsWith<IllegalStateException> {
            vectorizer().vectorizeAll(texts = listOf("a", "b"))
        }
    }

    @Test
    internal fun `reports reachability`() {
        assertTrue(actual = vectorizer().isReachable())

        status = 503
        assertFalse(actual = vectorizer().isReachable())
    }

    @Test
    internal fun `everything that changes a vector changes the fingerprint`() {
        val baseline = vectorizer().fingerprint()

        assertTrue(actual = vectorizer(config = config(model = "other")).fingerprint() != baseline)
        assertTrue(actual = vectorizer(config = config(revision = "rev-2")).fingerprint() != baseline)
        assertTrue(actual = vectorizer(config = config(prefix = "query: ")).fingerprint() != baseline)
        assertEquals(expected = "http-embedding", actual = baseline.kind)
        assertEquals(expected = 3, actual = baseline.dimension)
    }

    /**
     * Moving a service to another host must not invalidate every model trained against it, so the
     * URL is deliberately outside the fingerprint. The cost of that choice is documented on the
     * class: this cannot detect a model changed on the server.
     */
    @Test
    internal fun `the same model on a different host keeps its fingerprint`() {
        val here = vectorizer().fingerprint()
        val elsewhere = HttpEmbeddingVectorizer(
            config = config().copy(baseUrl = "http://another-host:1234/v1"),
        ).fingerprint()

        assertEquals(expected = here, actual = elsewhere)
    }

    @Test
    internal fun `rejects a configuration that cannot identify its model`() {
        assertFailsWith<IllegalArgumentException> { config().copy(modelRevision = " ") }
        assertFailsWith<IllegalArgumentException> { config().copy(baseUrl = "http://x/v1/") }
        assertFailsWith<IllegalArgumentException> { config().copy(batchSize = 0) }
        assertFailsWith<IllegalArgumentException> { config().copy(dimension = 0) }
    }
}
