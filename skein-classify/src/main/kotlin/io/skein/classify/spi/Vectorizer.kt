package io.skein.classify.spi

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.VectorizerCanary
import io.skein.classify.domain.VectorizerFingerprint

/**
 * Port for turning text into the features a classifier scores.
 *
 * The library ships its own featurisation — see
 * [io.skein.classify.application.HashingVectorizer] — and this port exists so an external
 * embedding model can be substituted for it without the rest of the pipeline noticing. A sparse
 * bag of hashed n-grams and a dense 384-dimensional sentence embedding are both just a
 * [FeatureVector], and the learner, the objective and the scoring loop work unchanged on either.
 *
 * Implementations must be safe to call from multiple threads.
 */
interface Vectorizer {

    /** Turns [text] into features. The same text must always produce the same vector. */
    fun vectorize(text: String): FeatureVector

    /**
     * Width of the feature space. Fixed for hashing and for embeddings; fitted for a vocabulary,
     * in which case it is only meaningful after fitting.
     */
    fun dimension(): Int

    /**
     * This vectorizer's identity, persisted with a model and verified on load. See
     * [VectorizerFingerprint] for why a mismatch must be fatal.
     */
    fun fingerprint(): VectorizerFingerprint

    /**
     * Fixed probe texts and the vectors this vectorizer currently produces for them, or `null` to
     * record nothing.
     *
     * A [VectorizerFingerprint] states what a vectorizer *is*; a [VectorizerCanary] samples what it
     * *does*. The two differ only when something outside the configuration can change the output —
     * which is the situation an external embedding service is permanently in, and which nothing in
     * its protocol reveals. Implementations whose output is fully determined by their configuration
     * gain nothing from this and should leave it at the default.
     *
     * Called during
     * [io.skein.classify.application.ModelStore.saveMultiLabel]; the result travels with the model
     * and is re-checked on load. Returning `null`, the default, disables the check entirely.
     */
    fun canary(): VectorizerCanary? {
        return null
    }
}
