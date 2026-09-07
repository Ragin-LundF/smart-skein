package io.skein.classify.domain

/**
 * Why a model scored [label] the way it did.
 *
 * The decomposition is exact and additive: `base` plus the contribution of **every** feature equals
 * [total], where [total] is the label's score minus the mean score across all labels. Because the
 * softmax is invariant to an additive shift, [total] is precisely the quantity that determines
 * [probability] — this is a decomposition of the real score, not an approximation of it.
 *
 * [contributions] holds only the highest-magnitude entries, so their sum is generally less than
 * `total - base`. Both signs are kept: evidence against a label is as informative as evidence for
 * it.
 *
 * A [probability] obtained straight from a [io.skein.classify.spi.Classifier] is uncalibrated.
 * [io.skein.classify.application.ClassificationService.explain] replaces it with the calibrated
 * value, so a service-level explanation always agrees with what `classify` reported.
 */
data class Explanation(
    val label: Label,
    val probability: Double,
    val base: Double,
    val total: Double,
    val contributions: List<FeatureContribution>,
)
