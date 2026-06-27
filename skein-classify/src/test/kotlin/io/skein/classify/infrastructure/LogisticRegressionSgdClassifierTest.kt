package io.skein.classify.infrastructure

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class LogisticRegressionSgdClassifierTest {

    private val vectorizer = HashingVectorizer(config = HashingConfig(key0 = 1L, key1 = 2L))

    private val trainingSet = listOf(
        vectorizer.vectorize(text = "win free money now") to Label(value = "spam"),
        vectorizer.vectorize(text = "free money win big") to Label(value = "spam"),
        vectorizer.vectorize(text = "team meeting at noon") to Label(value = "ham"),
        vectorizer.vectorize(text = "lunch and meeting today") to Label(value = "ham"),
    )

    private fun trained(epochs: Int = 50): LogisticRegressionSgdClassifier {
        val classifier = LogisticRegressionSgdClassifier()
        repeat(times = epochs) {
            trainingSet.forEach { (features, label) -> classifier.learn(features = features, label = label) }
        }
        return classifier
    }

    @Test
    internal fun `separates the two classes after several epochs of SGD`() {
        val classifier = trained()
        assertEquals(
            expected = Label(value = "spam"),
            actual = classifier.classify(features = vectorizer.vectorize(text = "win money")).label,
        )
        assertEquals(
            expected = Label(value = "ham"),
            actual = classifier.classify(features = vectorizer.vectorize(text = "meeting noon")).label,
        )
    }

    @Test
    internal fun `learning-rate decay shrinks the effective step over training`() {
        // With a strong decay the later steps barely move the weights, so the model ends up less
        // confident than one with a constant rate trained for the same number of steps.
        val constant = LogisticRegressionSgdClassifier(decayRate = 0.0)
        val decaying = LogisticRegressionSgdClassifier(decayRate = 50.0)
        repeat(times = 50) {
            trainingSet.forEach { (features, label) ->
                constant.learn(features = features, label = label)
                decaying.learn(features = features, label = label)
            }
        }
        val sample = vectorizer.vectorize(text = "win money")
        assertTrue(
            actual = constant.classify(features = sample).confidence > decaying.classify(features = sample).confidence,
            message = "decay must slow convergence relative to a constant rate",
        )
    }

    @Test
    internal fun `produces probabilities that sum to one`() {
        val prediction = trained().classify(features = vectorizer.vectorize(text = "win money"))
        assertEquals(
            expected = 1.0,
            actual = prediction.alternatives.sumOf { it.probability },
            absoluteTolerance = 1e-9,
        )
        assertTrue(actual = prediction.confidence in 0.0..1.0)
    }

    @Test
    internal fun `rejects classification before any training`() {
        assertFailsWith<IllegalStateException> {
            LogisticRegressionSgdClassifier().classify(features = vectorizer.vectorize(text = "anything"))
        }
    }

    @Test
    internal fun `forget clears learned labels`() {
        val classifier = trained(epochs = 1)
        classifier.forget()
        assertTrue(actual = classifier.labels().isEmpty())
    }
}
