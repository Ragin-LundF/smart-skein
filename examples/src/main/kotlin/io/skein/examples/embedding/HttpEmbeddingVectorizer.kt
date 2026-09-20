package io.skein.examples.embedding

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.spi.Vectorizer
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.security.MessageDigest
import java.time.Duration

/** Identifies this implementation in a [VectorizerFingerprint]. */
private const val KIND = "http-embedding"

private const val OK = 200

/**
 * A [Vectorizer] backed by an OpenAI-compatible `/embeddings` endpoint.
 *
 * Works unchanged against LM Studio, Ollama, llama.cpp's server, vLLM, Text Embeddings Inference and
 * the OpenAI API itself, because they all speak the same request shape. Depends on nothing beyond
 * the JDK's HTTP client and the JSON parser this module already uses, so it is meant to be copied
 * into your own codebase and adjusted rather than consumed as a library.
 *
 * ## The trade-off against running the model in-process
 *
 * This route is the fastest to get working: no model export, no native runtime, and swapping models
 * is a configuration change. What you give up is **verifiable model identity**.
 * `skein-classify-embedding-onnx` hashes the model file's bytes, so a model swapped in place is
 * refused at load. A service exposes no such thing — it returns vectors, and nothing in the protocol
 * says which weights produced them. [EmbeddingServiceConfig.modelRevision] is a label *you*
 * maintain, and it is only as trustworthy as your process for updating it.
 *
 * That matters because the failure is silent. A model updated on the server produces different
 * vectors for the same text, a model trained against the old ones keeps scoring without error, and
 * the labels are quietly wrong.
 *
 * ## Throughput
 *
 * Every call is a network round trip, so batching is not an optimisation here, it is the difference
 * between usable and not. [vectorizeAll] sends [EmbeddingServiceConfig.batchSize] inputs per
 * request; [vectorize] sends one and exists only because the port requires it.
 */
class HttpEmbeddingVectorizer(
    private val config: EmbeddingServiceConfig,
    private val client: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(config.timeoutSeconds))
        .build(),
) : Vectorizer {

    @Serializable
    private data class EmbeddingRequest(val model: String, val input: List<String>)

    @Serializable
    private data class EmbeddingEntry(val embedding: List<Float>, val index: Int = 0)

    @Serializable
    private data class EmbeddingResponse(val data: List<EmbeddingEntry>)

    private val json = Json { ignoreUnknownKeys = true }

    @Volatile
    private var resolvedDimension: Int? = config.dimension

    /** Embeds one text. Prefer [vectorizeAll]: this pays a whole round trip for a single record. */
    override fun vectorize(text: String): FeatureVector {
        return vectorizeAll(texts = listOf(text)).single()
    }

    /** Embeds [texts] in requests of [EmbeddingServiceConfig.batchSize], in the order given. */
    fun vectorizeAll(texts: List<String>): List<FeatureVector> {
        if (texts.isEmpty()) {
            return emptyList()
        }
        return texts.chunked(size = config.batchSize)
            .flatMap { chunk -> embed(texts = chunk) }
            .map { embedding -> FeatureVector(indices = IntArray(size = embedding.size) { it }, values = embedding) }
    }

    /**
     * The model's output width.
     *
     * Known up front when [EmbeddingServiceConfig.dimension] is set, and otherwise learned from the
     * first response — which means one probe request. Setting it explicitly is worth doing: it turns
     * a model swapped for one of a different width into an immediate, obvious failure instead of a
     * model that trains happily on the wrong feature space.
     */
    override fun dimension(): Int {
        resolvedDimension?.let { known -> return known }
        val probed = embed(texts = listOf("")).single().size
        resolvedDimension = probed
        return probed
    }

    /**
     * Identity as far as it can be established for a remote model.
     *
     * Covers the model name, the revision you declared, the input prefix and the width — everything
     * that changes the vector and that the client can actually see. The **base URL is deliberately
     * excluded**: the same model served from a different host must keep its fingerprint, or moving
     * a service between machines would invalidate every trained model for no reason.
     *
     * The consequence is the limitation named in the class documentation: this cannot detect a
     * changed model on the server. `skein-classify-embedding-onnx` can, because it hashes the file.
     */
    override fun fingerprint(): VectorizerFingerprint {
        val material = listOf(
            "model=${config.model}",
            "revision=${config.modelRevision}",
            "prefix=${config.inputPrefix}",
            "dimension=${dimension()}",
        ).joinToString(separator = "|")
        val digest = MessageDigest.getInstance("SHA-256").digest(material.toByteArray(Charsets.UTF_8))
        return VectorizerFingerprint(
            kind = KIND,
            dimension = dimension(),
            configDigest = digest.joinToString(separator = "") { byte -> "%02x".format(byte) },
        )
    }

    /** Whether the service answers. Use it to fail at startup rather than mid-training. */
    fun isReachable(): Boolean {
        return runCatching { embed(texts = listOf("ping")) }.isSuccess
    }

    private fun embed(texts: List<String>): List<FloatArray> {
        val payload = json.encodeToString(
            serializer = EmbeddingRequest.serializer(),
            value = EmbeddingRequest(
                model = config.model,
                input = texts.map { text -> config.inputPrefix + text },
            ),
        )
        val request = HttpRequest.newBuilder()
            .uri(URI.create(config.embeddingsUrl()))
            .timeout(Duration.ofSeconds(config.timeoutSeconds))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()

        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        check(value = response.statusCode() == OK) {
            "embedding request to ${config.embeddingsUrl()} failed with HTTP ${response.statusCode()}: " +
                response.body().take(n = 500)
        }
        val decoded = json.decodeFromString(deserializer = EmbeddingResponse.serializer(), string = response.body())
        check(value = decoded.data.size == texts.size) {
            "asked for ${texts.size} embeddings but the service returned ${decoded.data.size}"
        }
        // Sorted by index rather than trusted in arrival order: the OpenAI schema carries an index
        // precisely because a server is allowed to answer out of order, and a silently permuted
        // batch would attach every vector to the wrong record.
        return decoded.data
            .sortedBy { entry -> entry.index }
            .map { entry -> FloatArray(size = entry.embedding.size) { i -> entry.embedding[i] } }
    }
}
