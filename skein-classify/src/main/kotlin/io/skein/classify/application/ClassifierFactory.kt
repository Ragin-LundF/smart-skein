package io.skein.classify.application

import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import io.skein.classify.infrastructure.NaiveBayesClassifier
import io.skein.classify.spi.Classifier

/**
 * Builds a fresh, untrained [Classifier] for a persisted [ClassifierKindEnum] — the bridge from a
 * model file's recorded kind back to a runnable model.
 *
 * Instances use each classifier's default hyperparameters. Where you need tuned ones, pass your own
 * `() -> Classifier` lambda to [ModelEvaluator] instead of routing through here.
 *
 * The `when` is exhaustive with no `else`, so adding a classifier kind becomes a compile error at
 * every construction site rather than a silent fallback.
 */
object ClassifierFactory {

    fun create(kind: ClassifierKindEnum): Classifier {
        return when (kind) {
            ClassifierKindEnum.NAIVE_BAYES -> NaiveBayesClassifier()
            ClassifierKindEnum.LOGISTIC_REGRESSION -> LogisticRegressionSgdClassifier()
        }
    }
}
