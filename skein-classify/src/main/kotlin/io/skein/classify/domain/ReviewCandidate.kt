package io.skein.classify.domain

/**
 * A record the model is unsure about, surfaced for human labeling during active learning.
 *
 * @property record the unlabeled record
 * @property prediction the model's current best guess
 * @property margin top-two probability gap; smaller means more uncertain (more worth reviewing)
 */
data class ReviewCandidate(
    val record: Record,
    val prediction: Prediction,
    val margin: Double,
)
