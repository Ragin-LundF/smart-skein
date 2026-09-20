package io.skein.examples.embedding

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
 *   the served model; a stale value is how a silently-swapped model reaches production.
 * @property inputPrefix text prepended to every input. E5-family models are trained with
 *   `"passage: "` and `"query: "` markers and lose accuracy without them; most other families want
 *   `""`. Check the model card — this is the most common way an integration quietly underperforms.
 * @property dimension the model's output width, or `null` to learn it from the first response.
 * @property batchSize inputs per HTTP request.
 * @property timeoutSeconds per-request timeout.
 */
data class EmbeddingServiceConfig(
    val baseUrl: String,
    val model: String,
    val modelRevision: String,
    val inputPrefix: String = "",
    val dimension: Int? = null,
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
        require(value = batchSize > 0) { "batchSize must be positive, got $batchSize" }
        require(value = timeoutSeconds > 0) { "timeoutSeconds must be positive, got $timeoutSeconds" }
    }

    /** The embeddings endpoint this configuration points at. */
    fun embeddingsUrl(): String {
        return "$baseUrl/embeddings"
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
            )
        }
    }
}
