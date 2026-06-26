package io.skein.classify.domain

/**
 * How much of a learned record is persisted. There is intentionally **no default**: the caller
 * must choose explicitly, because the choice is a privacy decision, not a convenience.
 */
enum class PrivacyModeEnum {
    /**
     * Only the irreversible hashed feature vector is stored; the original record content is never
     * retained. Production guidance favors this mode.
     */
    FEATURES_ONLY,

    /**
     * The source record is also retained, encrypted at rest, to allow later re-mapping. Requires a
     * store that supports encryption (see `skein-store-postgres`); development guidance favors this.
     */
    ENCRYPTED_SOURCE,
}
