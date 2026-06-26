package io.skein.classify.infrastructure

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.spi.Classifier
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertNull

class ClassifierConcurrencyTest {

    private val vectorizer = HashingVectorizer(HashingConfig(key0 = 1L, key1 = 2L))
    private val sample = vectorizer.vectorize("rent payment for the apartment")

    private fun classifyWhileLearning(classifier: Classifier) {
        classifier.learn(vectorizer.vectorize("rent transfer landlord"), Label("housing"))
        val failure = AtomicReference<Throwable?>()
        val reader = thread {
            runCatching {
                repeat(2_000) { classifier.classify(sample) }
            }.onFailure { error -> failure.set(error) }
        }
        repeat(2_000) { index ->
            classifier.learn(vectorizer.vectorize("salary payout employer $index"), Label("income"))
        }
        reader.join()
        assertNull(failure.get(), message = "lock-free classify must not fail during concurrent training")
    }

    @Test
    fun `naive bayes classify is safe during concurrent training`() {
        classifyWhileLearning(NaiveBayesClassifier())
    }

    @Test
    fun `logistic regression classify is safe during concurrent training`() {
        classifyWhileLearning(LogisticRegressionSgdClassifier())
    }
}
