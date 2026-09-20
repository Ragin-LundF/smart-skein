package io.skein.classify.embedding.http.infrastructure

import io.skein.classify.embedding.http.spi.EmbeddingTransport
import io.skein.classify.embedding.http.spi.EmbeddingTransportResponse
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

/**
 * The default [EmbeddingTransport]: the JDK's own HTTP client and nothing else.
 *
 * Deliberately plain. It has no retry policy, no backoff and no circuit breaker, because there is
 * no answer to any of those that suits a local LM Studio and a rate-limited hosted API equally.
 * Supply your own [HttpClient] for a proxy, an executor or a TLS context, or implement
 * [EmbeddingTransport] outright to wrap whichever client your service already uses.
 */
class JdkHttpEmbeddingTransport(
    private val timeout: Duration,
    private val client: HttpClient = HttpClient.newBuilder().connectTimeout(timeout).build(),
) : EmbeddingTransport {

    override fun post(url: String, body: String, headers: Map<String, String>): EmbeddingTransportResponse {
        val builder = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(timeout)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
        headers.forEach { (name, value) -> builder.header(name, value) }
        val response = client.send(builder.build(), HttpResponse.BodyHandlers.ofString())
        return EmbeddingTransportResponse(statusCode = response.statusCode(), body = response.body())
    }
}
