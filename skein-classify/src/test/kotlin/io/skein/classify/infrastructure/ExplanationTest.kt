package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.spi.Classifier
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The same battery runs against both classifiers, because the decomposition is defined to have the
 * same shape for each.
 */
internal class ExplanationTest {

    private val tolerance = 1e-9
    private val labelA = Label(value = "A")
    private val labelB = Label(value = "B")

    /**
     * Feature 1 fires only for A, feature 2 only for B, and feature 9 fires equally for both —
     * feature 9 is the one whose contribution must vanish under centering.
     */
    private fun corpus(): List<LabeledFeatures> {
        return List(size = 12) {
            LabeledFeatures(
                label = labelA,
                features = FeatureVector(indices = intArrayOf(1, 9), values = floatArrayOf(1.0f, 1.0f)),
            )
        } + List(size = 12) {
            LabeledFeatures(
                label = labelB,
                features = FeatureVector(indices = intArrayOf(2, 9), values = floatArrayOf(1.0f, 1.0f)),
            )
        }
    }

    private val probe = FeatureVector(indices = intArrayOf(1, 2, 9), values = floatArrayOf(1.0f, 1.0f, 1.0f))

    private fun classifiers(): List<Classifier> {
        val naiveBayes = NaiveBayesClassifier()
        naiveBayes.learnAll(observations = corpus())
        val logistic = LogisticRegressionSgdClassifier()
        repeat(times = 60) { logistic.learnAll(observations = corpus()) }
        return listOf(naiveBayes, logistic)
    }

    @Test
    internal fun `base plus every contribution equals the centered total`() {
        classifiers().forEach { classifier ->
            val explanation = classifier.explain(features = probe, label = labelA, limit = Int.MAX_VALUE)!!
            val summed = explanation.base + explanation.contributions.sumOf { it.contribution }

            assertEquals(
                expected = explanation.total,
                actual = summed,
                absoluteTolerance = tolerance,
                message = "additivity broken for ${classifier::class.simpleName}",
            )
        }
    }

    @Test
    internal fun `the total equals the score minus the mean score across labels`() {
        classifiers().forEach { classifier ->
            val scores = classifier.logScores(features = probe)
            val mean = scores.values.average()
            val explanation = classifier.explain(features = probe, label = labelA, limit = Int.MAX_VALUE)!!

            assertEquals(
                expected = scores.getValue(key = labelA) - mean,
                actual = explanation.total,
                absoluteTolerance = 1e-6,
                message = "total mismatch for ${classifier::class.simpleName}",
            )
        }
    }

    @Test
    internal fun `a feature seen equally under every label is negligible next to a discriminating one`() {
        // This is the test that proves mean-centering: without it every Naive Bayes term is negative
        // and the commonest n-gram dominates every explanation. For Naive Bayes the cancellation is
        // algebraically exact; for SGD it is only as symmetric as training made the weights, so the
        // meaningful claim is relative magnitude.
        classifiers().forEach { classifier ->
            val explanation = classifier.explain(features = probe, label = labelA, limit = Int.MAX_VALUE)!!
            val shared = abs(x = explanation.contributions.first { it.featureIndex == 9 }.contribution)
            val discriminating = abs(x = explanation.contributions.first { it.featureIndex == 1 }.contribution)

            assertTrue(
                actual = shared < discriminating / 10.0,
                message = "shared=$shared vs discriminating=$discriminating for ${classifier::class.simpleName}",
            )
        }
    }

    @Test
    internal fun `Naive Bayes cancels a perfectly shared feature exactly`() {
        val naiveBayes = NaiveBayesClassifier()
        naiveBayes.learnAll(observations = corpus())
        val explanation = naiveBayes.explain(features = probe, label = labelA, limit = Int.MAX_VALUE)!!

        val shared = explanation.contributions.first { it.featureIndex == 9 }
        assertEquals(expected = 0.0, actual = shared.contribution, absoluteTolerance = 1e-12)
    }

    @Test
    internal fun `a discriminating feature helps its own label and hurts the other`() {
        classifiers().forEach { classifier ->
            val forA = classifier.explain(features = probe, label = labelA, limit = Int.MAX_VALUE)!!
            val forB = classifier.explain(features = probe, label = labelB, limit = Int.MAX_VALUE)!!

            val aFeatureUnderA = forA.contributions.first { it.featureIndex == 1 }.contribution
            val aFeatureUnderB = forB.contributions.first { it.featureIndex == 1 }.contribution

            assertTrue(actual = aFeatureUnderA > 0.0, message = "${classifier::class.simpleName}: $aFeatureUnderA")
            assertTrue(actual = aFeatureUnderB < 0.0, message = "${classifier::class.simpleName}: $aFeatureUnderB")
        }
    }

    @Test
    internal fun `contributions are ranked by absolute magnitude and truncated to the limit`() {
        classifiers().forEach { classifier ->
            val explanation = classifier.explain(features = probe, label = labelA, limit = 2)!!

            assertEquals(expected = 2, actual = explanation.contributions.size)
            val magnitudes = explanation.contributions.map { abs(x = it.contribution) }
            assertEquals(expected = magnitudes.sortedDescending(), actual = magnitudes)
        }
    }

    @Test
    internal fun `explanations carry no n-gram at the classifier level`() {
        classifiers().forEach { classifier ->
            val explanation = classifier.explain(features = probe, label = labelA, limit = 5)!!
            explanation.contributions.forEach { contribution ->
                assertEquals(expected = null, actual = contribution.ngram)
            }
        }
    }

    @Test
    internal fun `rejects a bad limit, an unknown label and an untrained model`() {
        classifiers().forEach { classifier ->
            assertFailsWith<IllegalArgumentException> {
                classifier.explain(features = probe, label = labelA, limit = 0)
            }
            assertFailsWith<IllegalArgumentException> {
                classifier.explain(features = probe, label = Label(value = "nope"), limit = 5)
            }
        }
        assertFailsWith<IllegalStateException> {
            NaiveBayesClassifier().explain(features = probe, label = labelA, limit = 5)
        }
    }
}
