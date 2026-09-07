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
        }
    }
}
