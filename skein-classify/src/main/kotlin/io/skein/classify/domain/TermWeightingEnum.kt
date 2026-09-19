package io.skein.classify.domain

/**
 * How a repeated n-gram is weighted in a [FeatureVector].
 *
 * A deliberate enum rather than a boolean: the choice changes what a feature value *means*, and a
 * `sublinear = true` at a call site says nothing about that.
 */
enum class TermWeightingEnum {

    /**
     * The number of times the n-gram occurred. Simple, and the right default: a term appearing
     * twice genuinely is stronger evidence than one appearing once.
     */
    RAW_COUNT,

    /**
     * `1 + ln(count)`, so the tenth occurrence adds far less than the second.
     *
     * Worth reaching for when record length varies a lot or when a repeated boilerplate token can
     * dominate a vector purely by frequency. Cheap, and usually a small improvement — but it
     * changes every feature value, so a model fitted under one weighting cannot be scored under the
     * other. The vectorizer fingerprint covers this, which is what turns that from a silent wrong
     * answer into a refused load.
     */
    SUBLINEAR,
}
