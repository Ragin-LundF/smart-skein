package io.skein.classify.domain

import java.security.SecureRandom

/**
 * Configuration for [io.skein.classify.application.HashingVectorizer].
 *
 * [key0]/[key1] are the keyed-hash secret: they make the feature hashing a keyed PRF, so the
 * mapping from text to feature indices is irreversible in aggregate and resistant to
 * hash-flooding. There is intentionally **no default key** — the caller must choose one explicitly,
 * because it is a privacy decision. Use [randomKey] for a fresh secret, or pass a fixed key to keep
 * feature indices stable across runs/processes (required when persisting or sharing a model).
 */
data class HashingConfig(
    val key0: Long,
    val key1: Long,
    val numFeatures: Int = DEFAULT_NUM_FEATURES,
    val charNgramMin: Int = DEFAULT_CHAR_NGRAM_MIN,
    val charNgramMax: Int = DEFAULT_CHAR_NGRAM_MAX,
    val wordNgramMin: Int = DEFAULT_WORD_NGRAM_MIN,
    val wordNgramMax: Int = DEFAULT_WORD_NGRAM_MAX,
) {

    init {
        require(numFeatures > 0) { "numFeatures must be positive" }
        require(charNgramMin in 1..charNgramMax) { "invalid char n-gram range" }
        require(wordNgramMin in 1..wordNgramMax) { "invalid word n-gram range" }
    }

    companion object {
        const val DEFAULT_NUM_FEATURES = 1 shl 18
        const val DEFAULT_CHAR_NGRAM_MIN = 3
        const val DEFAULT_CHAR_NGRAM_MAX = 5
        const val DEFAULT_WORD_NGRAM_MIN = 1
        const val DEFAULT_WORD_NGRAM_MAX = 2

        /**
         * Builds a config with a cryptographically random key. Note: feature indices then differ
         * per process, so a model trained with one random key cannot be reused with another — pass
         * a fixed, secret key when the model must be persisted or shared.
         */
        fun randomKey(): HashingConfig {
            val random = SecureRandom()
            return HashingConfig(key0 = random.nextLong(), key1 = random.nextLong())
        }
    }
}
