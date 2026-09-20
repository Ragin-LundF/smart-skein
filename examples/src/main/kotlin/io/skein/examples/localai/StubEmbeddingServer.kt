package io.skein.examples.localai

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import java.net.InetSocketAddress
import java.util.concurrent.atomic.AtomicInteger

/**
 * An in-process OpenAI-compatible `/embeddings` endpoint, for running this example and its tests
 * with nothing installed.
 *
 * **This is a development double, not something to deploy.** Its "embeddings" are a seeded random
 * projection of the input's tokens: deterministic, cheap, and carrying enough structure that a
 * linear model genuinely learns from them — but they are not semantic, so it cannot show you
 * whether embeddings beat hashing on your data. Point at a real model for that.
 *
 * What it *can* do, which a real service cannot be asked to do on demand, is **change its weights
 * while you watch**:
 *
 * - [seed] is the served model. Changing it is a model swap that no client can detect — the URL,
 *   the model name, the declared revision and the width are all unchanged, so the vectorizer's
 *   fingerprint still matches. This is precisely the failure a
 *   [io.skein.classify.domain.VectorizerCanary] exists to catch, and staging it against a real
 *   server means downloading a second model and restarting it.
 * - [scale] multiplies every vector, which is a service that has started L2-normalising its
 *   output. It rotates nothing, so a cosine-distance check sees nothing — while a linear
 *   classifier trained on the unnormalised vectors is broken.
 *
 * [requests] and [texts] count what crossed the wire, which is how the cost of batching is
 * measured rather than assumed.
 */
class StubEmbeddingServer(val dimension: Int = DEFAULT_DIMENSION) : AutoCloseable {

    /** The served "model". Change it to stage a swap. */
    @Volatile
    var seed: Long = 1L

    /** Multiplies every vector. Set to something other than 1 to stage a normalisation change. */
    @Volatile
    var scale: Float = 1.0f

    /** HTTP requests served. */
    val requests = AtomicInteger(0)

    /** Individual texts embedded, however they were batched. */
    val texts = AtomicInteger(0)

    private val json = Json { ignoreUnknownKeys = true }
    private var server: HttpServer? = null

    /** `http://localhost:<port>/v1`, valid only while the server is started. */
    lateinit var baseUrl: String
        private set

    /** Binds an ephemeral loopback port. */
    fun start(): StubEmbeddingServer {
        val started = HttpServer.create(InetSocketAddress(0), 0)
        started.createContext("/v1/embeddings") { exchange -> handle(exchange = exchange) }
        started.start()
        server = started
        baseUrl = "http://localhost:${started.address.port}/v1"
        return this
    }

    override fun close() {
        server?.stop(0)
        server = null
    }

    /** Zeroes the counters, so a measurement can start from a known point. */
    fun resetCounters() {
        requests.set(0)
        texts.set(0)
    }

    /** The same text under the same [seed] always gives the same vector. */
    fun embed(text: String): FloatArray {
        val vector = FloatArray(size = dimension)
        text.lowercase().split(TOKEN_BOUNDARY).filter { token -> token.isNotBlank() }.forEach { token ->
            var state = token.hashCode().toLong() * PRIME xor seed
            for (position in 0 until dimension) {
                state = state * MULTIPLIER + INCREMENT
                vector[position] += ((state ushr SHIFT).toInt() % RANGE) / RANGE.toFloat()
            }
        }
        for (position in vector.indices) {
            vector[position] *= scale
        }
        return vector
    }

    private fun handle(exchange: HttpExchange) {
        requests.incrementAndGet()
        val body = exchange.requestBody.readBytes().decodeToString()
        val inputs = (json.parseToJsonElement(body) as JsonObject)["input"]!!
            .jsonArray
            .map { element -> element.jsonPrimitive.content }
        texts.addAndGet(inputs.size)

        val entries = inputs.mapIndexed { index, text ->
            """{"index":$index,"embedding":[${embed(text = text).joinToString(separator = ",")}]}"""
        }
        val payload = """{"object":"list","data":[${entries.joinToString(separator = ",")}]}"""
        val bytes = payload.toByteArray()
        exchange.sendResponseHeaders(HTTP_OK, bytes.size.toLong())
        exchange.responseBody.use { stream -> stream.write(bytes) }
    }

    private companion object {
        const val DEFAULT_DIMENSION = 48
        const val HTTP_OK = 200
        const val PRIME = -0x61c8864680b583ebL
        const val MULTIPLIER = 6364136223846793005L
        const val INCREMENT = 1442695040888963407L
        const val SHIFT = 33
        const val RANGE = 1000
        val TOKEN_BOUNDARY = Regex(pattern = "\\W+")
    }
}
