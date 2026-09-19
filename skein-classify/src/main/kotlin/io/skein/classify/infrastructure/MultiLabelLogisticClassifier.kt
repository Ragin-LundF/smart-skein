package io.skein.classify.infrastructure

import io.skein.classify.domain.ClassifierHyperparameters
import io.skein.classify.domain.Explanation
import io.skein.classify.domain.FeatureContribution
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabelPredictionFactory
import io.skein.classify.spi.MultiLabelClassifier
import kotlin.math.abs

/**
 * A fitted one-vs-rest logistic model: one independent linear head per label over shared
 * [MultiLabelWeights].
 *
 * Immutable and thread-safe. Produced by [LbfgsMultiLabelLearner] or restored by
 * [io.skein.classify.application.ModelStore]; there is no incremental update, because the optimiser
 * that fits it needs the whole corpus — see [io.skein.classify.spi.BatchLearner].
 */
class MultiLabelLogisticClassifier(
    private val weights: MultiLabelWeights,
    private val tuning: ClassifierHyperparameters = ClassifierHyperparameters.DEFAULTS,
) : MultiLabelClassifier {

    private val indexByLabel: Map<Label, Int> = weights.labels
        .withIndex()
        .associate { (index, label) -> label to index }

    /** The fitted weights, for persistence and inspection. */
    fun weights(): MultiLabelWeights {
        return weights
    }

    override fun logits(features: FeatureVector): Map<Label, Double> {
        val scores = weights.logits(features = features)
        return weights.labels.withIndex().associate { (index, label) -> label to scores[index] }
    }

    override fun labels(): Set<Label> {
        return indexByLabel.keys
    }

    /**
     * Decomposes [label]'s logit into per-feature contributions.
     *
     * Unlike the single-label explanation, nothing is centered across labels: each head is an
     * independent linear model, so `intercept + sum(w_f * x_f)` *is* the logit, and the
     * decomposition is exact rather than a shift-invariant stand-in. `base` is the intercept and
     * `total` is the logit itself.
     */
    override fun explain(features: FeatureVector, label: Label, limit: Int): Explanation? {
        require(value = limit > 0) { "limit must be positive, got $limit" }
        val labelIndex = indexByLabel[label] ?: return null
        val base = weights.intercepts[labelIndex]
        var total = base
        val contributions = ArrayList<FeatureContribution>(features.indices.size)
        for (position in features.indices.indices) {
            val feature = features.indices[position]
            val value = features.values[position]
            val contribution = weights.weightOf(feature = feature, labelIndex = labelIndex) * value
            if (contribution == 0.0) {
                continue
            }
            total += contribution
            contributions.add(
                element = FeatureContribution(
                    featureIndex = feature,
                    featureValue = value,
                    contribution = contribution,
                ),
            )
        }
        contributions.sortByDescending { entry -> abs(x = entry.contribution) }
        return Explanation(
            label = label,
            probability = MultiLabelPredictionFactory.logistic(logit = total),
            base = base,
            total = total,
            contributions = contributions.take(n = limit),
        )
    }

    override fun hyperparameters(): ClassifierHyperparameters {
        return tuning
    }
}
