package io.skein.classify.domain

/**
 * Per-class quality for one [label].
 *
 * [precision] is `tp / (tp + fp)`, [recall] is `tp / (tp + fn)` and [f1] their harmonic mean; each
 * is `0.0` — never `NaN` — when its denominator is zero, so a report is always printable and
 * averageable. [support] is how many times [label] was the ground truth.
 */
data class LabelMetrics(
    val label: Label,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val support: Int,
)
