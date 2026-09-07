package io.skein.classify.domain

/**
 * Whether an explanation resolves hashed buckets back to the n-grams that produced them.
 *
 * Privacy: [WITH_NGRAMS] hands clear-text fragments of the explained record back to the caller. It
 * builds no stored index and weakens no saved model — the n-grams are re-derived on the call from
 * text the caller already holds, so it grants nothing to an adversary who does not already have
 * both the hashing key and the record. What it changes is the *result*: an explanation in this mode
 * contains source text, so do not log it, cache it, or send it anywhere the record itself may not
 * go.
 */
enum class AttributionModeEnum {

    /** Default. Contributions carry only the opaque bucket id, and are safe to log. */
    BUCKETS_ONLY,

    /** Opt-in. Contributions additionally carry a representative source n-gram in clear text. */
    WITH_NGRAMS,
}
