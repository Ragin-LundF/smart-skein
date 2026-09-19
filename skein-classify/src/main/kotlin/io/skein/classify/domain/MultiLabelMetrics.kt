package io.skein.classify.domain

/**
 * What a multi-label model scored on a set of evaluated cases.
 *
 * Conventions, which differ from [EvaluationReport] in ways that matter:
 * - A [ConfusionMatrix] **does not apply**. With co-occurring labels there is no single predicted
 *   class to confuse with a single true class; every (case, label) pair is its own independent
 *   binary decision. Do not reach for one.
 * - [micro] pools true/false positives and negatives over every (case, label) pair before dividing,
 *   so frequent labels dominate it. It answers "how often is a label assignment right?".
 * - [macro] averages the per-label figures with equal weight, so a label with four examples counts
 *   as much as one with four hundred. It answers "how well is the model doing across the
 *   taxonomy?", and on a long-tailed corpus it is always the lower and more honest number.
 * - [macro] averages over **every** label in [perLabel], including labels that were predicted but
 *   never true. A hallucinated label drags it down, matching [EvaluationReport]'s convention.
 * - [exactMatchRatio] is the fraction of cases whose predicted set equals the expected set
 *   exactly — the strictest reading, and it counts a case with one label missing as a total miss.
 * - [coverage] is the fraction of cases that received at least one label. Falling coverage as the
 *   threshold rises is the cost side of abstention, and it is invisible in precision alone.
 */
data class MultiLabelMetrics(
    val sampleCount: Int,
    val threshold: Double,
    val micro: AveragedMetrics,
    val macro: AveragedMetrics,
    val exactMatchRatio: Double,
    val coverage: Double,
    val perLabel: List<LabelMetrics>,
) {

    /** Metrics for [label], or `null` when it was neither expected nor predicted anywhere. */
    fun metricsFor(label: Label): LabelMetrics? {
        return perLabel.firstOrNull { metrics -> metrics.label == label }
    }
}
