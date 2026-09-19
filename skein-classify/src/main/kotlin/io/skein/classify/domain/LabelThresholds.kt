package io.skein.classify.domain

/**
 * The acceptance threshold each label is held to.
 *
 * One global threshold assumes every label's scores are calibrated the same way, and they are not.
 * A label with four hundred examples produces confident, well-spread probabilities; a label with
 * four produces a timid cluster that never reaches 0.5, so a global cut silences it entirely while
 * a well-supported label with the same cut is happily over-firing. Per-label thresholds let each
 * head be read on its own terms.
 *
 * This is a *decision* layer, not a model change — it re-reads scores the model already produced,
 * so changing a threshold costs nothing and needs no refit. Fit thresholds with
 * [io.skein.classify.application.ThresholdOptimizer], and fit them on data the model was not
 * trained on.
 *
 * [fallback] applies to any label without an entry, which is what a model scored against a
 * threshold set fitted before that label existed will hit.
 */
class LabelThresholds(private val byLabel: Map<Label, Double>, val fallback: Double) {

    init {
        require(value = fallback in 0.0..1.0) { "fallback must be between 0.0 and 1.0, got $fallback" }
        require(value = byLabel.values.all { threshold -> threshold in 0.0..1.0 }) {
            "every threshold must be between 0.0 and 1.0"
        }
    }

    /** The threshold for [label], or [fallback] when none was fitted. */
    fun of(label: Label): Double {
        return byLabel[label] ?: fallback
    }

    /** Labels with a threshold of their own. */
    fun tunedLabels(): Set<Label> {
        return byLabel.keys
    }

    /** The thresholds as a map, for persistence and reporting. */
    fun asMap(): Map<Label, Double> {
        return byLabel.toMap()
    }

    /**
     * The labels of [prediction] that clear their own threshold.
     *
     * Use this in place of [MultiLabelPrediction.labels], which applies the single threshold the
     * prediction carries.
     */
    fun accepted(prediction: MultiLabelPrediction): Set<Label> {
        return prediction.ranked
            .filter { scored -> scored.probability >= of(label = scored.label) }
            .map { scored -> scored.label }
            .toSet()
    }

    companion object {

        /** One threshold for every label — the behaviour of a plain global cut. */
        fun uniform(threshold: Double): LabelThresholds {
            return LabelThresholds(byLabel = emptyMap(), fallback = threshold)
        }
    }
}
