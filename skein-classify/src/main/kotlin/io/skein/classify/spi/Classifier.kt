package io.skein.classify.spi

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Prediction

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

    /** Discards all learned state. */
    fun forget()
}
