package io.skein.classify.domain

/**
 * One hashed feature's signed additive contribution to a label's centered score.
 *
 * [featureIndex] is an irreversible SipHash bucket, not a word. It is a **stable pseudonym**: for a
 * fixed hashing key the same bucket always means the same n-gram, which is enough to diff two model
 * versions, watch a feature's influence drift, or spot a single bucket dominating a prediction —
 * all without revealing any text. Bucket ids are safe to log.
 *
 * [ngram] is `null` unless the caller explicitly asked for [AttributionModeEnum.WITH_NGRAMS], in
 * which case it is one representative n-gram, taken from the record being explained, that hashed
 * into this bucket. Other n-grams in the same record may share the bucket, and [contribution] is
 * the sum over all of them. A non-null [ngram] is source text — treat it as such.
 */
data class FeatureContribution(
    val featureIndex: Int,
    val featureValue: Float,
    val contribution: Double,
    val ngram: String? = null,
)
