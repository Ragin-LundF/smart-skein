package io.skein.classify.embedding.http.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.VectorizerCanary
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.spi.EmbeddingTransport
import io.skein.classify.spi.BatchVectorizer
import java.security.MessageDigest
import java.time.Duration

/** Identifies this implementation in a [VectorizerFingerprint]. */
private const val KIND = "http-embedding"

private const val OK = 200

/**
 * A [io.skein.classify.spi.Vectorizer] backed by an OpenAI-compatible `/embeddings` endpoint.
 *
 * Works unchanged against LM Studio, Ollama, llama.cpp's server, vLLM, Text Embeddings Inference
 * and the OpenAI API itself, because they all speak the same request shape.
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
 * **Set [EmbeddingServiceConfig.canaryProbes] and the failure stops being silent.** The probes are
 * embedded when the model is saved and re-embedded when it is loaded; a model changed on the server
 * moves them, and the load throws instead of scoring. It is the only check here that looks at the
 * vectors themselves rather than at what the service claims. See [VectorizerCanary].
 *
 * ## Throughput
 *
 * Every call is a network round trip, so batching is not an optimisation here, it is the difference
 * between usable and not. [vectorizeAll] sends [EmbeddingServiceConfig.batchSize] inputs per
 * request; [vectorize] sends one and exists only because the port requires it.
 */
class HttpEmbeddingVectorizer(
    private val config: EmbeddingServiceConfig,
    private val transport: EmbeddingTransport = JdkHttpEmbeddingTransport(
        timeout = Duration.ofSeconds(config.timeoutSeconds),
    ),
) : BatchVectorizer {

    @Volatile
    private var resolvedDimension: Int? = config.dimension

    @Volatile
    private var capturedCanary: VectorizerCanary? = null

    /** Embeds one text. Prefer [vectorizeAll]: this pays a whole round trip for a single record. */
    override fun vectorize(text: String): FeatureVector {
        return vectorizeAll(texts = listOf(text)).single()
    }

    /** Embeds [texts] in requests of [EmbeddingServiceConfig.batchSize], in the order given. */
    override fun vectorizeAll(texts: List<String>): List<FeatureVector> {
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
     * that changes the vector and that the client can actually see. The **base URL and the request
     * headers are deliberately excluded**: the same model served from a different host, or reached
     * with a rotated token, must keep its fingerprint, or moving a service between machines would
     * invalidate every trained model for no reason.
     *
     * The consequence is the limitation named in the class documentation: this cannot detect a
     * changed model on the server. [canary] is what can.
     */
    override fun fingerprint(): VectorizerFingerprint {
        val width = dimension()
        val material = listOf(
            "model=${config.model}",
            "revision=${config.modelRevision}",
            "prefix=${config.inputPrefix}",
            "dimension=$width",
        ).joinToString(separator = "|")
        val digest = MessageDigest.getInstance("SHA-256").digest(material.toByteArray(Charsets.UTF_8))
        return VectorizerFingerprint(
            kind = KIND,
            dimension = width,
            configDigest = digest.joinToString(separator = "") { byte -> "%02x".format(byte) },
        )
    }

    /**
     * The configured probes and the vectors the service currently returns for them, or `null` when
     * [EmbeddingServiceConfig.canaryProbes] is empty.
     *
     * Captured once and cached, so saving a model does not re-embed the probes a second time.
     */
    override fun canary(): VectorizerCanary? {
        if (config.canaryProbes.isEmpty()) {
            return null
        }
        capturedCanary?.let { known -> return known }
        // One probe per request, never batched: batch composition perturbs floating-point
        // accumulation order, and a reference captured in a batch of three compared against one
        // re-embedded alone would drift for a reason that has nothing to do with the model.
        val references = config.canaryProbes.map { probe -> embed(texts = listOf(probe)).single() }
        val captured = VectorizerCanary(
            probes = config.canaryProbes,
            references = references,
            tolerance = config.canaryTolerance,
        )
        capturedCanary = captured
        return captured
    }

    /** Whether the service answers. Use it to fail at startup rather than mid-training. */
    fun isReachable(): Boolean {
        return runCatching { embed(texts = listOf("ping")) }.isSuccess
    }

    private fun embed(texts: List<String>): List<FloatArray> {
        val payload = OpenAiEmbeddingProtocol.encodeRequest(
            model = config.model,
            inputs = texts.map { text -> config.inputPrefix + text },
        )
        val response = transport.post(url = config.embeddingsUrl(), body = payload, headers = config.headers)
        check(value = response.statusCode == OK) {
            "embedding request to ${config.embeddingsUrl()} failed with HTTP ${response.statusCode}: " +
                response.body.take(n = BODY_EXCERPT)
        }
        return OpenAiEmbeddingProtocol.decodeResponse(
            body = response.body,
            expectedCount = texts.size,
            expectedWidth = resolvedDimension,
        )
    }

    private companion object {
        private const val BODY_EXCERPT = 500
    }
}
