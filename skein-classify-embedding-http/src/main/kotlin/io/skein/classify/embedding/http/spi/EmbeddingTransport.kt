package io.skein.classify.embedding.http.spi

/**
 * Port for the one part of this adapter that needs a real external system: sending a JSON body
 * somewhere and getting one back.
 *
 * Isolating it has two purposes. The obvious one is testing — every error path in the vectorizer
 * (a non-200, a malformed body, a short batch, a drifted vector) is reachable through a fake
 * transport, without standing up an HTTP server for each.
 *
 * The other is a boundary on what this module promises. Published as a library, an HTTP client
 * attracts requests for retry policies, circuit breakers, connection pooling and proxy support,
 * none of which have one right answer. This module owns the **protocol**; the transport is yours,
 * and [io.skein.classify.embedding.http.infrastructure.JdkHttpEmbeddingTransport] is a working
 * default rather than the only option.
 */
interface EmbeddingTransport {

    /**
     * POSTs [body] as `application/json` to [url] and returns the response.
     *
     * Implementations must not translate a transport failure into a successful-looking response:
     * a timeout or a refused connection must throw. Distinguishing "the service is down" from
     * "the model changed" depends on it.
     */
    fun post(url: String, body: String, headers: Map<String, String>): EmbeddingTransportResponse
}

/** An HTTP status and body, which is all the protocol layer needs. */
class EmbeddingTransportResponse(val statusCode: Int, val body: String)
