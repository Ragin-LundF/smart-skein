package io.skein.classify.embedding.http.infrastructure

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json

/**
 * The OpenAI `/embeddings` wire format: the request body, and how to read a response back.
 *
 * LM Studio, Ollama, llama.cpp's server, vLLM, Text Embeddings Inference and the OpenAI API all
 * speak this, which is the whole reason one adapter covers them.
 */
internal object OpenAiEmbeddingProtocol {

    private val json = Json { ignoreUnknownKeys = true }

    @Serializable
    private data class EmbeddingRequest(val model: String, val input: List<String>)

    @Serializable
    private data class EmbeddingEntry(val embedding: List<Float>, val index: Int = 0)

    @Serializable
    private data class EmbeddingResponse(val data: List<EmbeddingEntry>)

    /** Encodes a request body for [model] over [inputs], already prefixed. */
    fun encodeRequest(model: String, inputs: List<String>): String {
        return json.encodeToString(
            serializer = EmbeddingRequest.serializer(),
            value = EmbeddingRequest(model = model, input = inputs),
        )
    }

    /**
     * Reads [body] into one vector per input, in the order the inputs were sent.
     *
     * Entries are **sorted by index rather than trusted in arrival order**: the OpenAI schema
     * carries an index precisely because a server may answer out of order, and a silently permuted
     * batch attaches every vector to the wrong record.
     *
     * @param expectedCount how many inputs were sent, checked against what came back.
     * @param expectedWidth the width every vector must have, or `null` when it is not yet known.
     */
    fun decodeResponse(body: String, expectedCount: Int, expectedWidth: Int?): List<FloatArray> {
        val decoded = try {
            json.decodeFromString(deserializer = EmbeddingResponse.serializer(), string = body)
        } catch (cause: SerializationException) {
            throw IllegalStateException(
                "the embedding service returned a body that is not an OpenAI embeddings response: " +
                    body.take(n = BODY_EXCERPT),
                cause,
            )
        }
        check(value = decoded.data.size == expectedCount) {
            "asked for $expectedCount embeddings but the service returned ${decoded.data.size}"
        }
        val vectors = decoded.data
            .sortedBy { entry -> entry.index }
            .map { entry -> FloatArray(size = entry.embedding.size) { position -> entry.embedding[position] } }
        if (expectedWidth != null) {
            // Without this a service switched from 384 to 768 dimensions is silently absorbed:
            // scoring drops every feature index past the end of the weight matrix and keeps the
            // rest, which come from a different model. Confident, wrong labels -- the exact
            // failure this adapter's fingerprint and canary exist to prevent.
            val wrong = vectors.firstOrNull { vector -> vector.size != expectedWidth }
            check(value = wrong == null) {
                "the embedding service returned ${wrong?.size}-dimension vectors but this model " +
                    "expects $expectedWidth; the served model has been changed for one of a " +
                    "different width"
            }
        }
        return vectors
    }

    private const val BODY_EXCERPT = 500
}
