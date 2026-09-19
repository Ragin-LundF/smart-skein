package io.skein.classify.application

import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import io.skein.classify.infrastructure.NaiveBayesClassifier
import kotlin.test.Test
import kotlin.test.assertFailsWith
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
    internal fun `every kind either builds an untrained classifier or is refused by name`() {
        ClassifierKindEnum.entries.forEach { kind ->
            // An exhaustive `when` **expression**, so a newly added kind is a compile error here
            // rather than quietly inheriting whichever branch happens to come last.
            val checked: Unit = when (kind) {
                ClassifierKindEnum.NAIVE_BAYES,
                ClassifierKindEnum.LOGISTIC_REGRESSION,
                -> assertTrue(actual = ClassifierFactory.create(kind = kind).labels().isEmpty())

                // Not a Classifier at all: it implements MultiLabelClassifier and is fitted in one
                // batch, so there is nothing untrained to hand back.
                ClassifierKindEnum.MULTI_LABEL_LOGISTIC -> {
                    val failure = assertFailsWith<IllegalArgumentException> {
                        ClassifierFactory.create(kind = kind)
                    }
                    assertTrue(actual = failure.message!!.contains(other = "loadMultiLabel"))
                }
            }
            checked
        }
    }

    @Test
    internal fun `returns a fresh instance on every call`() {
        val first = ClassifierFactory.create(kind = ClassifierKindEnum.NAIVE_BAYES)
        val second = ClassifierFactory.create(kind = ClassifierKindEnum.NAIVE_BAYES)
        assertTrue(actual = first !== second)
    }
}
