package io.skein.classify.infrastructure

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NaiveBayesClassifierTest {

    private val vectorizer = HashingVectorizer(HashingConfig(key0 = 1L, key1 = 2L))

    private fun trained(): NaiveBayesClassifier {
        val classifier = NaiveBayesClassifier()
        classifier.learn(vectorizer.vectorize("win free money now"), Label("spam"))
        classifier.learn(vectorizer.vectorize("free money win big"), Label("spam"))
        classifier.learn(vectorizer.vectorize("team meeting at noon"), Label("ham"))
        classifier.learn(vectorizer.vectorize("lunch and meeting today"), Label("ham"))
        return classifier
    }

    @Test
    fun `predicts the class whose training text it resembles`() {
        val classifier = trained()
        assertEquals(expected = Label("spam"), actual = classifier.classify(vectorizer.vectorize("win money")).label)
        assertEquals(expected = Label("ham"), actual = classifier.classify(vectorizer.vectorize("meeting noon")).label)
    }

    @Test
    fun `produces calibrated probabilities that rank and sum to one`() {
        val prediction = trained().classify(vectorizer.vectorize("win money"))
        assertTrue(prediction.confidence in 0.0..1.0)
        assertEquals(expected = prediction.confidence, actual = prediction.alternatives.first().probability)
        assertEquals(
            expected = 1.0,
            actual = prediction.alternatives.sumOf { it.probability },
            absoluteTolerance = 1e-9,
        )
        val sortedDesc = prediction.alternatives.sortedByDescending { it.probability }
        assertEquals(expected = sortedDesc, actual = prediction.alternatives)
    }

    @Test
    fun `rejects classification before any training`() {
        assertFailsWith<IllegalStateException> {
            NaiveBayesClassifier().classify(vectorizer.vectorize("anything"))
        }
    }

    @Test
    fun `forget clears learned labels`() {
        val classifier = trained()
        classifier.forget()
        assertTrue(classifier.labels().isEmpty())
    }
}
