package io.skein.classify.domain

/** Aggregate view of what an engine has learned: total observations and per-label counts. */
data class ClassificationMetrics(
    val totalObservations: Int,
    val perLabelCounts: Map<Label, Int>,
)
