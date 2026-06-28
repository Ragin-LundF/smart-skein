package io.skein.classify.infrastructure

import io.skein.classify.application.HashingVectorizer
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class NaiveBayesClassifierTest {

    private val vectorizer = HashingVectorizer(config = HashingConfig(key0 = 1L, key1 = 2L))

    private fun trained(): NaiveBayesClassifier {
        val classifier = NaiveBayesClassifier()
        classifier.learn(features = vectorizer.vectorize(text = "win free money now"), label = Label(value = "spam"))
        classifier.learn(features = vectorizer.vectorize(text = "free money win big"), label = Label(value = "spam"))
        classifier.learn(features = vectorizer.vectorize(text = "team meeting at noon"), label = Label(value = "ham"))
        classifier.learn(
            features = vectorizer.vectorize(text = "lunch and meeting today"),
            label = Label(value = "ham"),
        )
        return classifier
    }

    @Test
    internal fun `predicts the class whose training text it resembles`() {
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
    internal fun `produces calibrated probabilities that rank and sum to one`() {
        val prediction = trained().classify(features = vectorizer.vectorize(text = "win money"))
        assertTrue(actual = prediction.confidence in 0.0..1.0)
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
    internal fun `rejects classification before any training`() {
        assertFailsWith<IllegalStateException> {
            NaiveBayesClassifier().classify(features = vectorizer.vectorize(text = "anything"))
        }
    }

    @Test
    internal fun `forget clears learned labels`() {
        val classifier = trained()
        classifier.forget()
        assertTrue(actual = classifier.labels().isEmpty())
    }

    @Test
    internal fun `learnAll produces identical predictions to sequential learn`() {
        val observations = listOf(
            LabeledFeatures(label = Label("spam"), features = vectorizer.vectorize("win free money now")),
            LabeledFeatures(label = Label("spam"), features = vectorizer.vectorize("free money win big")),
            LabeledFeatures(label = Label("ham"), features = vectorizer.vectorize("team meeting at noon")),
            LabeledFeatures(label = Label("ham"), features = vectorizer.vectorize("lunch and meeting today")),
        )
        val probe = vectorizer.vectorize(text = "win money")

        val sequential = NaiveBayesClassifier()
        observations.forEach { obs -> sequential.learn(features = obs.features, label = obs.label) }

        val batch = NaiveBayesClassifier()
        batch.learnAll(observations = observations)

        val seqPred = sequential.classify(features = probe)
        val batchPred = batch.classify(features = probe)
        assertEquals(expected = seqPred.label, actual = batchPred.label)
        assertEquals(expected = seqPred.confidence, actual = batchPred.confidence, absoluteTolerance = 1e-9)
    }

    @Test
    internal fun `learnAll rejects classification when observation list is empty`() {
        val classifier = NaiveBayesClassifier()
        assertFailsWith<IllegalStateException> {
            classifier.learnAll(observations = emptyList())
            classifier.classify(features = vectorizer.vectorize("anything"))
        }
    }
}
