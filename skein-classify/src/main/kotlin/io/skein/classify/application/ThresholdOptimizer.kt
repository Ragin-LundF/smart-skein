package io.skein.classify.application

import io.skein.classify.domain.Label
import io.skein.classify.domain.LabelThresholds
import io.skein.classify.domain.MultiLabelOutcome

/** Threshold assigned to a label whose scores give nothing to optimise. */
private const val DEFAULT_FALLBACK = 0.5

/**
 * Fits a per-label acceptance threshold by maximising each label's own F1.
 *
 * **Fit this on held-out data.** Thresholds fitted on the rows a model was trained on sit exactly
 * where that model's training-set scores happen to fall, which is optimistic by construction — the
 * same trap as fitting a temperature on training data, and just as silent. Cross-validated outcomes
 * from [MultiLabelCrossValidator] are a good source: every row there was scored by a model that had
 * not seen it.
 *
 * Each label is optimised independently, which is the point: the heads are independent, so there is
 * nothing to trade off between them.
 */
class ThresholdOptimizer(private val fallback: Double = DEFAULT_FALLBACK) {

    init {
        require(value = fallback in 0.0..1.0) { "fallback must be between 0.0 and 1.0, got $fallback" }
    }

    /**
     * The threshold maximising F1 for each label that occurs in [outcomes].
     *
     * A label with no positive examples has no F1 to maximise — every threshold scores zero — and
     * receives [fallback] rather than an arbitrary winner of a tie.
     */
    fun fit(outcomes: List<MultiLabelOutcome>): LabelThresholds {
        require(value = outcomes.isNotEmpty()) { "cannot fit thresholds on an empty outcome list" }
        val labels = sortedSetOf<String>()
        outcomes.forEach { outcome ->
            outcome.expected.forEach { label -> labels.add(element = label.value) }
            outcome.prediction.ranked.forEach { scored -> labels.add(element = scored.label.value) }
        }
        val tuned = labels.mapNotNull { value ->
            val label = Label(value = value)
            bestThreshold(outcomes = outcomes, label = label)?.let { threshold -> label to threshold }
        }.toMap()
        return LabelThresholds(byLabel = tuned, fallback = fallback)
    }

    /**
     * The threshold maximising [label]'s F1, or `null` when the label has no positives.
     *
     * A single descending sweep rather than a search over sampled candidates. F1 only changes where
     * the accepted set changes, and the accepted set only changes at an observed score, so walking
     * the scores from high to low and accepting one group at a time visits **every** distinct
     * outcome exactly once. That makes it exact — sampling candidates instead can step straight
     * over the one threshold that separates the classes cleanly and settle for a worse cut — and it
     * is `O(n log n)` rather than `O(n^2)`, which matters at a million rows times a large taxonomy.
     *
     * Walking downwards also gives the documented tie-break for free: the strict comparison keeps
     * the first threshold to reach the best score, and that is the highest, so of two cuts that
     * score identically the more conservative one wins.
     */
    private fun bestThreshold(outcomes: List<MultiLabelOutcome>, label: Label): Double? {
        val scored = outcomes.map { outcome ->
            outcome.prediction.probabilityOf(label = label) to (label in outcome.expected)
        }.sortedByDescending { (probability, _) -> probability }
        val positives = scored.count { (_, expected) -> expected }
        if (positives == 0) {
            return null
        }

        var truePositives = 0
        var falsePositives = 0
        var bestF1 = -1.0
        var bestThreshold = fallback
        var index = 0
        while (index < scored.size) {
            val probability = scored[index].first
            // Accept every case sharing this score before measuring: a threshold cannot separate
            // two cases that scored identically, so they stand or fall together.
            while (index < scored.size && scored[index].first == probability) {
                if (scored[index].second) truePositives++ else falsePositives++
                index++
            }
            val f1 = f1Of(
                truePositives = truePositives,
                falsePositives = falsePositives,
                falseNegatives = positives - truePositives,
            )
            if (f1 > bestF1) {
                bestF1 = f1
                bestThreshold = probability
            }
        }
        return bestThreshold
    }

    private fun f1Of(truePositives: Int, falsePositives: Int, falseNegatives: Int): Double {
        val denominator = 2 * truePositives + falsePositives + falseNegatives
        return if (denominator == 0) 0.0 else 2.0 * truePositives / denominator
    }

}
