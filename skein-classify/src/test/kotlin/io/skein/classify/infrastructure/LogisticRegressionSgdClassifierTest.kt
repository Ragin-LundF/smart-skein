package io.skein.classify.infrastructure

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LogisticRegressionSgdClassifierTest {

    private val vectorizer = HashingVectorizer(HashingConfig(key0 = 1L, key1 = 2L))

    private val trainingSet = listOf(
        vectorizer.vectorize("win free money now") to Label("spam"),
        vectorizer.vectorize("free money win big") to Label("spam"),
        vectorizer.vectorize("team meeting at noon") to Label("ham"),
        vectorizer.vectorize("lunch and meeting today") to Label("ham"),
    )

    private fun trained(epochs: Int = 50): LogisticRegressionSgdClassifier {
        val classifier = LogisticRegressionSgdClassifier()
        repeat(epochs) {
            trainingSet.forEach { (features, label) -> classifier.learn(features, label) }
        }
        return classifier
    }

    @Test
    fun `separates the two classes after several epochs of SGD`() {
        val classifier = trained()
        assertEquals(expected = Label("spam"), actual = classifier.classify(vectorizer.vectorize("win money")).label)
        assertEquals(expected = Label("ham"), actual = classifier.classify(vectorizer.vectorize("meeting noon")).label)
    }

    @Test
    fun `learning-rate decay shrinks the effective step over training`() {
        // With a strong decay the later steps barely move the weights, so the model ends up less
        // confident than one with a constant rate trained for the same number of steps.
        val constant = LogisticRegressionSgdClassifier(decayRate = 0.0)
        val decaying = LogisticRegressionSgdClassifier(decayRate = 50.0)
        repeat(50) {
            trainingSet.forEach { (features, label) ->
                constant.learn(features, label)
                decaying.learn(features, label)
            }
        }
        val sample = vectorizer.vectorize("win money")
        assertTrue(
            constant.classify(sample).confidence > decaying.classify(sample).confidence,
            message = "decay must slow convergence relative to a constant rate",
        )
    }

    @Test
    fun `produces probabilities that sum to one`() {
        val prediction = trained().classify(vectorizer.vectorize("win money"))
        assertEquals(
            expected = 1.0,
            actual = prediction.alternatives.sumOf { it.probability },
            absoluteTolerance = 1e-9,
        )
        assertTrue(prediction.confidence in 0.0..1.0)
    }

    @Test
    fun `rejects classification before any training`() {
        assertFailsWith<IllegalStateException> {
            LogisticRegressionSgdClassifier().classify(vectorizer.vectorize("anything"))
        }
    }

    @Test
    fun `forget clears learned labels`() {
        val classifier = trained(epochs = 1)
        classifier.forget()
        assertTrue(classifier.labels().isEmpty())
    }
}
