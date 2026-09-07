package io.skein.classify.spi

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.ClassifierHyperparameters
import io.skein.classify.domain.Explanation
import io.skein.classify.domain.Prediction
import kotlin.math.ln

/**
 * Port for a learnable classifier. Implementations (Naive Bayes here, logistic regression later)
 * are swappable strategies; all support incremental learning so the model improves observation by
 * observation without a full retrain.
 */
interface Classifier {

    /** Incrementally updates the model with one labeled observation. */
    fun learn(features: FeatureVector, label: Label)

    /**
     * Learns from many observations in one call. Implementations may override this for a faster
     * batch path (e.g. Naive Bayes publishes a single snapshot instead of one per observation).
     * The default is a simple loop over [learn].
     */
    fun learnAll(observations: List<LabeledFeatures>) {
        observations.forEach { obs -> learn(features = obs.features, label = obs.label) }
    }

    /** Predicts the most likely label for [features]. Requires at least one prior observation. */
    fun classify(features: FeatureVector): Prediction

    /** Labels the model has learned so far. */
    fun labels(): Set<Label>

    /**
     * Raw per-label scores in log or logit space, before the softmax that [classify] applies. This
     * is the input to [io.skein.classify.domain.PredictionFactory.fromLogScores] and to calibration
     * fitting. Only the differences between labels matter — the softmax is shift-invariant — so an
     * implementation may return scores carrying any common offset.
     *
     * The default derives scores from [classify] by taking `ln(probability)`, which is exact up to
     * that common offset.
     *
     * ponytail: a probability that underflows to zero makes the default return negative infinity,
     * and Naive Bayes underflows routinely (a score gap beyond ~745 nats is ordinary across
     * hundreds of features). Ceiling: the default is unusable for calibrating a confident model.
     * Upgrade path: override it, as both classifiers shipped here do.
     */
    fun logScores(features: FeatureVector): Map<Label, Double> {
        return classify(features = features).alternatives
            .associate { scored -> scored.label to ln(x = scored.probability) }
    }

    /**
     * Ranked per-feature contributions to [label]'s score, largest absolute contribution first and
     * truncated to [limit] entries. Returns `null` when this implementation cannot attribute its
     * score to individual features, which is the default; both classifiers shipped here override it.
     *
     * The reported probability is this classifier's own **uncalibrated** value.
     */
    fun explain(features: FeatureVector, label: Label, limit: Int): Explanation? {
        return null
    }

    /**
     * The tuning this classifier was constructed with, so it can be persisted with a model and
     * restored alongside it. The default reports [ClassifierHyperparameters.DEFAULTS]; both
     * classifiers shipped here override it.
     */
    fun hyperparameters(): ClassifierHyperparameters {
        return ClassifierHyperparameters.DEFAULTS
    }

    /** Discards all learned state. */
    fun forget()
}
