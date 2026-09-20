package io.skein.examples.localai

import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.infrastructure.HttpEmbeddingVectorizer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

/** An embedding model that was found *and* proven to answer, with the width it returns. */
data class DiscoveredModel(val id: String, val dimension: Int)

/**
 * Finds a usable embedding model on an OpenAI-compatible server.
 *
 * ## Why this is not one line
 *
 * The obvious approach — hard-code the model identifier — does not survive contact with a real
 * server, for three reasons this class exists to handle:
 *
 * 1. **The identifier is not guessable.** It is whatever the server chose when the model was
 *    downloaded. The same `multilingual-e5-small` weights are `multilingual-e5-small-mlx` on a
 *    machine that pulled the MLX conversion. A wrong id is an HTTP 400.
 * 2. **`/v1/models` lists what is *downloaded*, not what is *loaded*.** Unless just-in-time
 *    loading is enabled, an advertised id can still answer `"No models loaded"`.
 * 3. **Most advertised models are not embedding models.** A chat model answers `/v1/chat`, not
 *    `/v1/embeddings`, and the only reliable way to tell them apart is to ask for an embedding.
 *
 * So the only trustworthy test of "can I embed with this" is to try, which is what
 * [firstEmbeddingModel] does.
 *
 * ## Using this in production
 *
 * **Discover once at startup, then pin.** Call [firstEmbeddingModel] when your service boots, log
 * what it found, and put that id in configuration — do not re-discover per request. Once a model
 * has trained a classifier, its identity is part of your deployment: pin the id, bump
 * [EmbeddingServiceConfig.modelRevision] when you change it, and let the
 * [io.skein.classify.domain.VectorizerCanary] catch the case where someone changes it and forgets.
 *
 * Discovery is a convenience for getting started and for failing clearly at boot. It is not a
 * substitute for knowing which model you deployed.
 */
object ModelDiscovery {

    private val json = Json { ignoreUnknownKeys = true }

    /** Model ids the server advertises, or empty when it does not answer at all. */
    fun advertised(baseUrl: String, client: HttpClient = defaultClient()): List<String> {
        val request = HttpRequest.newBuilder()
            .uri(URI.create("$baseUrl/models"))
            .timeout(PROBE_TIMEOUT)
            .GET()
            .build()
        return runCatching {
            val body = client.send(request, HttpResponse.BodyHandlers.ofString()).body()
            ((json.parseToJsonElement(body) as JsonObject)["data"] as JsonArray)
                .map { entry -> (entry as JsonObject)["id"]!!.jsonPrimitive.content }
        }.getOrDefault(defaultValue = emptyList())
    }

    /**
     * The first advertised model that actually returns a vector, trying [preferred] first.
     *
     * Returns `null` when the server does not answer, has nothing loaded, or advertises only
     * models that cannot embed. Use [diagnose] to tell a caller which of those it was.
     *
     * Each candidate costs one embedding request, so this is a startup cost, not a hot path.
     */
    fun firstEmbeddingModel(
        baseUrl: String,
        preferred: String? = null,
        client: HttpClient = defaultClient(),
    ): DiscoveredModel? {
        val candidates = buildList {
            preferred?.let { add(it) }
            addAll(advertised(baseUrl = baseUrl, client = client).filter { id -> id != preferred })
        }
        return candidates.firstNotNullOfOrNull { id ->
            runCatching {
                val probe = HttpEmbeddingVectorizer(
                    config = EmbeddingServiceConfig(
                        baseUrl = baseUrl,
                        model = id,
                        modelRevision = "discovery-probe",
                        timeoutSeconds = PROBE_TIMEOUT_SECONDS,
                    ),
                )
                DiscoveredModel(id = id, dimension = probe.dimension())
            }.getOrNull()
        }
    }

    /**
     * A human-readable explanation of why [firstEmbeddingModel] found nothing, written to be
     * printed straight at an operator.
     */
    fun diagnose(baseUrl: String, client: HttpClient = defaultClient()): String {
        val listed = advertised(baseUrl = baseUrl, client = client)
        if (listed.isEmpty()) {
            return """
                |No OpenAI-compatible server answered at $baseUrl.
                |  LM Studio: Developer tab -> Start Server (default port 1234).
                |  Ollama:    ollama serve            (then baseUrl = http://localhost:11434/v1)
            """.trimMargin()
        }
        return """
            |$baseUrl answers and advertises ${listed.size} model(s), but none of them embedded.
            |  Advertised: ${listed.joinToString()}
            |
            |  /v1/models lists what is DOWNLOADED, not what is LOADED. Load an embedding model:
            |      lms load <id> -y        (lms ps shows what is actually resident)
            |  A chat model cannot embed -- pick one built for embeddings.
        """.trimMargin()
    }

    private fun defaultClient(): HttpClient {
        return HttpClient.newBuilder().connectTimeout(CONNECT_TIMEOUT).build()
    }

    private val CONNECT_TIMEOUT: Duration = Duration.ofSeconds(3)
    private val PROBE_TIMEOUT: Duration = Duration.ofSeconds(5)
    private const val PROBE_TIMEOUT_SECONDS = 120L
}
