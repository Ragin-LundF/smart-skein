package io.skein.classify.domain

/**
 * Result of classifying a record: the winning [label], its [confidence] (the top probability),
 * and the full ranked list of [alternatives] (highest probability first, including the winner).
 */
data class Prediction(
    val label: Label,
    val confidence: Double,
    val alternatives: List<ScoredLabel>,
)
