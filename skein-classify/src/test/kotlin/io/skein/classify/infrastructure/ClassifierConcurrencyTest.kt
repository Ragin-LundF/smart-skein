package io.skein.classify.infrastructure

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.spi.Classifier
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertNull

internal class ClassifierConcurrencyTest {

    private val vectorizer = HashingVectorizer(config = HashingConfig(key0 = 1L, key1 = 2L))
    private val sample = vectorizer.vectorize(text = "rent payment for the apartment")

    private fun classifyWhileLearning(classifier: Classifier) {
        classifier.learn(
            features = vectorizer.vectorize(text = "rent transfer landlord"),
            label = Label(value = "housing"),
        )
        val failure = AtomicReference<Throwable?>()
        val reader = thread {
            runCatching {
                repeat(times = 2_000) { classifier.classify(features = sample) }
            }.onFailure { error -> failure.set(error) }
        }
        repeat(times = 2_000) { index ->
            classifier.learn(
                features = vectorizer.vectorize(text = "salary payout employer $index"),
                label = Label(value = "income"),
            )
        }
        reader.join()
        assertNull(actual = failure.get(), message = "lock-free classify must not fail during concurrent training")
    }

    @Test
    internal fun `naive bayes classify is safe during concurrent training`() {
        classifyWhileLearning(classifier = NaiveBayesClassifier())
    }

    @Test
    internal fun `logistic regression classify is safe during concurrent training`() {
        classifyWhileLearning(classifier = LogisticRegressionSgdClassifier())
    }
}
