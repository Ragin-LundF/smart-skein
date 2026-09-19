package io.skein.classify.application

import io.skein.classify.domain.ClassifierHyperparameters
import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import io.skein.classify.infrastructure.NaiveBayesClassifier
import io.skein.classify.spi.Classifier

/**
 * Builds a fresh, untrained [Classifier] for a persisted [ClassifierKindEnum] — the bridge from a
 * model file's recorded kind back to a runnable model.
 *
 * The single-argument overload uses each classifier's defaults; pass [ClassifierHyperparameters] to
 * rebuild a tuned model, which is what restoring a saved model does.
 *
 * The `when` is exhaustive with no `else`, so adding a classifier kind becomes a compile error at
 * every construction site rather than a silent fallback.
 *
 * Not every kind yields a [Classifier]: [ClassifierKindEnum.MULTI_LABEL_LOGISTIC] implements
 * [io.skein.classify.spi.MultiLabelClassifier] instead and is rejected here by name.
 */
object ClassifierFactory {

    fun create(kind: ClassifierKindEnum): Classifier {
        return create(kind = kind, hyperparameters = ClassifierHyperparameters.DEFAULTS)
    }

    fun create(kind: ClassifierKindEnum, hyperparameters: ClassifierHyperparameters): Classifier {
        return when (kind) {
            ClassifierKindEnum.NAIVE_BAYES -> NaiveBayesClassifier(
                smoothingAlpha = hyperparameters.smoothingAlpha,
            )

            ClassifierKindEnum.LOGISTIC_REGRESSION -> LogisticRegressionSgdClassifier(
                initialLearningRate = hyperparameters.initialLearningRate,
                decayRate = hyperparameters.decayRate,
                l2Regularization = hyperparameters.l2Regularization,
            )

            // A multi-label model implements a different port: it assigns any number of labels, is
            // fitted in one batch, and has no `learn(one)` to offer. There is no Classifier to
            // return here, and quietly substituting a single-label one would be worse than failing.
            ClassifierKindEnum.MULTI_LABEL_LOGISTIC -> throw IllegalArgumentException(
                "$kind is a MultiLabelClassifier, not a Classifier; " +
                    "load it with ModelStore.loadMultiLabel or fit one with LbfgsMultiLabelLearner",
            )
        }
    }
}
