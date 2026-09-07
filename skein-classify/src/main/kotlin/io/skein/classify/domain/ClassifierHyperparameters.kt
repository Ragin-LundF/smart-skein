package io.skein.classify.domain

/**
 * The tuning knobs of a [io.skein.classify.spi.Classifier], in one record so they can be persisted
 * alongside a model and restored with it.
 *
 * Each field is only meaningful to the classifiers that use it — [smoothingAlpha] to Naive Bayes,
 * the three SGD fields to logistic regression — and the rest stay at their defaults. This mirrors
 * [HashingConfig], which likewise gathers one component's tuning into a single value.
 *
 * Without persisting these, a model tuned away from the defaults comes back from disk as a
 * *different model*, because loading replays the stored observations through a freshly constructed
 * classifier.
 */
data class ClassifierHyperparameters(
    val smoothingAlpha: Double = DEFAULT_SMOOTHING_ALPHA,
    val initialLearningRate: Double = DEFAULT_LEARNING_RATE,
    val decayRate: Double = DEFAULT_DECAY_RATE,
    val l2Regularization: Double = DEFAULT_L2_REGULARIZATION,
) {

    init {
        require(value = smoothingAlpha > 0.0) { "smoothingAlpha must be positive" }
        require(value = initialLearningRate > 0.0) { "initialLearningRate must be positive" }
        require(value = decayRate >= 0.0) { "decayRate must not be negative" }
        require(value = l2Regularization >= 0.0) { "l2Regularization must not be negative" }
    }

    companion object {

        /** Additive (Laplace) smoothing weight for Naive Bayes. */
        const val DEFAULT_SMOOTHING_ALPHA = 1.0

        /** Base SGD step size for logistic regression. */
        const val DEFAULT_LEARNING_RATE = 0.1

        /** SGD step decay; zero keeps the step size constant. */
        const val DEFAULT_DECAY_RATE = 0.0

        /** L2 shrinkage; zero disables regularization. */
        const val DEFAULT_L2_REGULARIZATION = 0.0

        /** The values every classifier uses when the caller does not tune it. */
        val DEFAULTS = ClassifierHyperparameters()
    }
}
