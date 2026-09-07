package io.skein.classify.application

import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import io.skein.classify.infrastructure.NaiveBayesClassifier
import kotlin.test.Test
import kotlin.test.assertTrue

internal class ClassifierFactoryTest {

    @Test
    internal fun `builds the classifier recorded by each kind`() {
        assertTrue(actual = ClassifierFactory.create(kind = ClassifierKindEnum.NAIVE_BAYES) is NaiveBayesClassifier)
        assertTrue(
            actual = ClassifierFactory.create(
                kind = ClassifierKindEnum.LOGISTIC_REGRESSION,
            ) is LogisticRegressionSgdClassifier,
        )
    }

    @Test
    internal fun `builds an untrained classifier for every kind`() {
        // Iterating the entries means a newly added kind fails here rather than silently.
        ClassifierKindEnum.entries.forEach { kind ->
            assertTrue(actual = ClassifierFactory.create(kind = kind).labels().isEmpty())
        }
    }

    @Test
    internal fun `returns a fresh instance on every call`() {
        val first = ClassifierFactory.create(kind = ClassifierKindEnum.NAIVE_BAYES)
        val second = ClassifierFactory.create(kind = ClassifierKindEnum.NAIVE_BAYES)
        assertTrue(actual = first !== second)
    }
}
