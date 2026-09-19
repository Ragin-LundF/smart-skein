package io.skein.classify.spi

import io.skein.classify.domain.MultiLabeledFeatures

/**
 * Port for fitting a [MultiLabelClassifier] from a whole corpus at once.
 *
 * Separate from [Classifier] on purpose. [Classifier.learn] takes one observation, which is what
 * makes incremental and active learning work and is exactly what a batch optimiser cannot offer:
 * L-BFGS needs the full design matrix to compute a gradient at all, so there is no honest
 * `learn(one)` to implement. Widening [Classifier] would have forced a meaningless method onto it.
 *
 * The two are complementary rather than competing — online SGD when observations arrive one at a
 * time, a batch learner when the corpus is in hand and the best possible fit is wanted.
 */
interface BatchLearner {

    /**
     * Fits a model over [observations] and returns it. Implementations must not mutate
     * [observations], so the same corpus can be fitted repeatedly with different tuning.
     */
    fun fit(observations: List<MultiLabeledFeatures>): MultiLabelClassifier
}
