package io.skein.classify.domain

import kotlin.math.exp

/**
 * Turns raw per-label logits into a [MultiLabelPrediction]. Shared by every multi-label classifier
 * so one sigmoid convention and one ordering rule are applied everywhere.
 *
 * The multi-label sibling of [PredictionFactory], and deliberately *not* a softmax: each label gets
 * its own independent logistic transform, so the probabilities never compete.
 */
object MultiLabelPredictionFactory {

    /**
     * Applies the logistic function to each of [logits] and ranks the results, highest first, ties
     * broken by label value so the order is deterministic.
     *
     * The sigmoid is evaluated on the non-overflowing branch: `exp` of a large positive argument
     * overflows to infinity, and a margin beyond ~700 is ordinary once a few hundred features
     * agree, so the naive `1 / (1 + exp(-z))` returns `NaN` on exactly the confident cases.
     */
    fun fromLogits(logits: Map<Label, Double>, threshold: Double): MultiLabelPrediction {
        val ranked = logits.entries
            .map { entry -> ScoredLabel(label = entry.key, probability = logistic(logit = entry.value)) }
            .sortedWith(
                comparator = compareByDescending<ScoredLabel> { scored -> scored.probability }
                    .thenBy { scored -> scored.label.value },
            )
        return MultiLabelPrediction(ranked = ranked, threshold = threshold)
    }

    /** The logistic function, computed on whichever branch cannot overflow. */
    fun logistic(logit: Double): Double {
        if (logit >= 0.0) {
            return 1.0 / (1.0 + exp(x = -logit))
        }
        val z = exp(x = logit)
        return z / (1.0 + z)
    }
}
