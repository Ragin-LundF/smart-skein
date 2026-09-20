package io.skein.classify.embedding.http.domain

import io.skein.classify.domain.VectorizerCanary

/**
 * Everything needed to talk to an OpenAI-compatible embedding endpoint, and everything that has to
 * be pinned so a model can be trusted not to have changed underneath you.
 *
 * @property baseUrl root of the service, without a trailing slash — LM Studio serves
 *   `http://localhost:1234/v1`, Ollama `http://localhost:11434/v1`.
 * @property model the identifier the service knows the model by, sent in every request.
 * @property modelRevision **your** label for the exact weights behind [model]. An external service
 *   can be updated without changing the model name, and nothing in the protocol reveals it, so this
 *   is the only handle on which weights produced a vector. Change it whenever you change or update
 *   the served model; a stale value is how a silently-swapped model reaches production. It is a
 *   declaration, not a verification — [canaryProbes] is the part that actually checks.
 * @property inputPrefix text prepended to every input. E5-family models are trained with
 *   `"passage: "` and `"query: "` markers and lose accuracy without them; most other families want
 *   `""`. Check the model card — this is the most common way an integration quietly underperforms.
 * @property dimension the model's output width, or `null` to learn it from the first response.
 *   Setting it is worth doing: it makes a model swapped for one of a different width fail
 *   immediately instead of training happily on the wrong feature space.
 * @property headers extra request headers — `Authorization: Bearer …` for a hosted API, a proxy
 *   token, a tracing id. **Excluded from the fingerprint**, for the same reason [baseUrl] is: a
 *   rotated credential must not invalidate every model you have trained. Nothing that changes the
 *   vectors may go in here.
 * @property canaryProbes texts to embed at save time and re-embed on load, catching a model
 *   changed on the server. Empty disables the check. See [VectorizerCanary] for why this exists
 *   and [EmbeddingProbes.DEFAULT] for a starting set; the probes are stored in the model file in
 *   clear text, so they must never come from your corpus.
 * @property canaryTolerance largest relative L2 drift still accepted as the same model.
 * @property batchSize inputs per HTTP request.
 * @property timeoutSeconds per-request timeout.
 */
data class EmbeddingServiceConfig(
    val baseUrl: String,
    val model: String,
    val modelRevision: String,
    val inputPrefix: String = "",
    val dimension: Int? = null,
    val headers: Map<String, String> = emptyMap(),
    val canaryProbes: List<String> = emptyList(),
    val canaryTolerance: Double = VectorizerCanary.DEFAULT_TOLERANCE,
    val batchSize: Int = DEFAULT_BATCH_SIZE,
    val timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS,
) {

    init {
        require(value = baseUrl.isNotBlank()) { "baseUrl must not be blank" }
        require(value = !baseUrl.endsWith(suffix = "/")) { "baseUrl must not end with '/', got '$baseUrl'" }
        require(value = model.isNotBlank()) { "model must not be blank" }
        require(value = modelRevision.isNotBlank()) {
            "modelRevision must not be blank: it is the only record of which weights produced a vector"
        }
        require(value = dimension == null || dimension > 0) { "dimension must be positive when given" }
        require(value = headers.keys.none { name -> name.isBlank() }) { "a header name must not be blank" }
        require(value = canaryProbes.distinct().size == canaryProbes.size) { "canary probes must be distinct" }
        require(value = batchSize > 0) { "batchSize must be positive, got $batchSize" }
        require(value = timeoutSeconds > 0) { "timeoutSeconds must be positive, got $timeoutSeconds" }
    }

    /** The embeddings endpoint this configuration points at. */
    fun embeddingsUrl(): String {
        return "$baseUrl/embeddings"
    }

    /**
     * Redacts header **values**.
     *
     * A generated `toString` would print a bearer token into the first log line or assertion
     * failure that touched this object, which is not a risk the copy-it-yourself version had to
     * think about and a published one does.
     */
    override fun toString(): String {
        val redacted = headers.keys.sorted().joinToString(prefix = "{", postfix = "}") { name -> "$name=***" }
        return "EmbeddingServiceConfig(baseUrl=$baseUrl, model=$model, modelRevision=$modelRevision, " +
            "inputPrefix=$inputPrefix, dimension=$dimension, headers=$redacted, " +
            "canaryProbes=${canaryProbes.size}, canaryTolerance=$canaryTolerance, " +
            "batchSize=$batchSize, timeoutSeconds=$timeoutSeconds)"
    }

    companion object {
        /**
         * Large enough to amortise the per-request overhead, small enough to stay inside the
         * default request-size limits of the common local servers.
         */
        const val DEFAULT_BATCH_SIZE = 32
        const val DEFAULT_TIMEOUT_SECONDS = 120L

        /**
         * LM Studio's default local server, with a small multilingual model.
         *
         * `multilingual-e5-small` covers around 100 languages at 384 dimensions and roughly half a
         * gigabyte on disk, which is the practical sweet spot for running locally on a CPU. Its
         * `"passage: "` prefix is mandatory, not decorative.
         */
        fun lmStudioMultilingualE5Small(): EmbeddingServiceConfig {
            return EmbeddingServiceConfig(
                baseUrl = "http://localhost:1234/v1",
                model = "text-embedding-multilingual-e5-small",
                modelRevision = "lmstudio/multilingual-e5-small@1",
                inputPrefix = "passage: ",
                dimension = 384,
                canaryProbes = EmbeddingProbes.DEFAULT,
            )
        }
    }
}
