package io.skein.classify.domain

/**
 * One bucket of a reliability diagram: every outcome whose confidence fell in
 * `[lowerBound, upperBound)` — the topmost bin includes `1.0`.
 *
 * A well-calibrated model has [accuracy] close to [meanConfidence] in every bin. Both are `0.0`
 * for an empty bin; empty bins are retained so the diagram has a fixed axis and a CSV export stays
 * rectangular.
 */
data class CalibrationBin(
    val lowerBound: Double,
    val upperBound: Double,
    val count: Int,
    val meanConfidence: Double,
    val accuracy: Double,
)
