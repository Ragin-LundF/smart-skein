package io.skein.classify.domain

/**
 * Result of scoring a record against labels that may co-occur.
 *
 * Unlike [Prediction], the probabilities in [ranked] are **independent sigmoids** — one per
 * one-vs-rest head. They do not compete and they do not sum to one, so two labels at 0.9 is an
 * ordinary outcome and so is every label below the threshold. Reading these as a distribution is
 * the single most common way to misuse multi-label output.
 *
 * [ranked] holds every label the model knows, highest probability first, ties broken by label value
 * so the order is deterministic for a given model rather than dependent on map iteration.
 *
 * [threshold] is the cut applied by [labels]. It is a business decision, not a technical default:
 * see the precision/recall table produced by
 * [io.skein.classify.application.MultiLabelEvaluator.sweep].
 */
data class MultiLabelPrediction(val ranked: List<ScoredLabel>, val threshold: Double) {

    init {
        require(value = threshold in 0.0..1.0) { "threshold must be between 0.0 and 1.0, got $threshold" }
    }

    /** The labels at or above [threshold]. Empty when the model is confident about none of them. */
    fun labels(): Set<Label> {
        return ranked.filter { scored -> scored.probability >= threshold }
            .map { scored -> scored.label }
            .toSet()
    }

    /**
     * The [count] highest-scoring labels regardless of [threshold], or all of them when the model
     * knows fewer.
     *
     * Worth preferring over [labels] for human review: measured on the reference corpus the correct
     * labels sit in the top 3 far more often (recall@3 = 0.775) than they win outright
     * (recall@1 = 0.484), so a reviewer shown three candidates is right far more often than one
     * shown the winner alone.
     */
    fun topK(count: Int): List<ScoredLabel> {
        require(value = count > 0) { "count must be positive, got $count" }
        return ranked.take(n = count)
    }

    /** The same scores read at a different cut. Re-scores nothing, so a sweep is free. */
    fun at(threshold: Double): MultiLabelPrediction {
        return MultiLabelPrediction(ranked = ranked, threshold = threshold)
    }

    /** The probability assigned to [label], or `0.0` when this model does not know it. */
    fun probabilityOf(label: Label): Double {
        return ranked.firstOrNull { scored -> scored.label == label }?.probability ?: 0.0
    }
}
