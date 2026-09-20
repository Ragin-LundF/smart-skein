package io.skein.classify.embedding.onnx.application

/** Entries retained before the least recently used is evicted. */
private const val DEFAULT_MAX_ENTRIES = 50_000

/** `LinkedHashMap`'s documented default, restated because the access-order constructor needs it. */
private const val LOAD_FACTOR = 0.75f

/**
 * A bounded, least-recently-used cache of pooled embeddings.
 *
 * **Why this is worth having.** Embedding is the expensive step by a wide margin — a transformer
 * encoder is roughly 1–3 ms per record against 41 µs for hashed n-grams, so a million records is
 * over half an hour of inference. A hyperparameter sweep re-embeds the same corpus for every
 * candidate, and cross-validation re-embeds the training rows of every fold. With a cache, all of
 * that is paid once.
 *
 * Keys must include the vectorizer's fingerprint, so a changed model or pooling strategy cannot be
 * served a vector computed under the old one. [OnnxEmbeddingVectorizer][
 * io.skein.classify.embedding.onnx.infrastructure.OnnxEmbeddingVectorizer] builds them that way.
 *
 * Bounded because the alternative is a map that grows with the corpus, which is the one thing a
 * pipeline built for a million records must not do. Deduplicating the corpus first — see
 * `CorpusDeduplicator` — makes the cache far smaller than the corpus, and often small enough to
 * hold all of it.
 *
 * Safe to use from several threads.
 */
class EmbeddingCache(private val maxEntries: Int = DEFAULT_MAX_ENTRIES) {

    init {
        require(value = maxEntries > 0) { "maxEntries must be positive, got $maxEntries" }
    }

    private val entries = object : LinkedHashMap<String, FloatArray>(maxEntries, LOAD_FACTOR, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, FloatArray>?): Boolean {
            return size > this@EmbeddingCache.maxEntries
        }
    }

    /** The cached embedding for [key], or `null`. Counts as an access for eviction purposes. */
    fun get(key: String): FloatArray? {
        synchronized(lock = entries) {
            return entries[key]
        }
    }

    /** Stores [embedding] under [key], evicting the least recently used entry if full. */
    fun put(key: String, embedding: FloatArray) {
        synchronized(lock = entries) {
            entries[key] = embedding
        }
    }

    /** Entries currently held. */
    fun size(): Int {
        synchronized(lock = entries) {
            return entries.size
        }
    }

    fun clear() {
        synchronized(lock = entries) {
            entries.clear()
        }
    }
}
