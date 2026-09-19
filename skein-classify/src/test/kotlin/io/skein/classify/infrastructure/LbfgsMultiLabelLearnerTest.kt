package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

internal class LbfgsMultiLabelLearnerTest {

    private val featureCount = 8
    private val alpha = Label(value = "ALPHA")
    private val beta = Label(value = "BETA")

    private fun features(vararg indices: Int): FeatureVector {
        return FeatureVector(
            indices = indices.sortedArray(),
            values = FloatArray(size = indices.size) { 1.0f },
        )
    }

    private fun observation(vararg indices: Int, labels: Set<Label>): MultiLabeledFeatures {
        return MultiLabeledFeatures(features = features(indices = indices), labels = labels)
    }

    /**
     * Feature 0 marks ALPHA, feature 1 marks BETA, and a record can carry both. Feature 2 is an
     * explicit negative: it belongs to neither label, which is what teaches the heads where their
     * label does *not* apply.
     */
    private fun toyCorpus(): List<MultiLabeledFeatures> {
        return listOf(
            observation(0, 4, labels = setOf(alpha)),
            observation(0, 5, labels = setOf(alpha)),
            observation(1, 4, labels = setOf(beta)),
            observation(1, 6, labels = setOf(beta)),
            observation(0, 1, labels = setOf(alpha, beta)),
            observation(2, 7, labels = emptySet()),
        )
    }

    private fun learner(keepFraction: Double = 1.0): LbfgsMultiLabelLearner {
        return LbfgsMultiLabelLearner(
            featureCount = featureCount,
            inverseRegularization = 10.0,
            gradientTolerance = 1e-8,
            keepFraction = keepFraction,
        )
    }

    @Test
    internal fun `separates a toy corpus it was fitted on`() {
        val model = learner().fit(observations = toyCorpus())

        val alphaOnly = model.predict(features = features(0, 4), threshold = 0.5)
        assertEquals(expected = setOf(alpha), actual = alphaOnly.labels())

        val betaOnly = model.predict(features = features(1, 6), threshold = 0.5)
        assertEquals(expected = setOf(beta), actual = betaOnly.labels())
    }

    /**
     * The capability the whole port exists for. A softmax model cannot produce this: its scores are
     * constrained to sum to one, so a genuine second label is suppressed by construction.
     */
    @Test
    internal fun `assigns both labels to a record that carries both`() {
        val model = learner().fit(observations = toyCorpus())

        val prediction = model.predict(features = features(0, 1), threshold = 0.5)

        assertEquals(expected = setOf(alpha, beta), actual = prediction.labels())
        assertTrue(
            actual = prediction.ranked.sumOf { scored -> scored.probability } > 1.0,
            message = "independent sigmoids should not sum to one",
        )
    }

    @Test
    internal fun `assigns no label to a record matching an explicit negative`() {
        val model = learner().fit(observations = toyCorpus())

        val prediction = model.predict(features = features(2, 7), threshold = 0.5)

        assertTrue(
            actual = prediction.labels().isEmpty(),
            message = "expected no labels, got ${prediction.ranked}",
        )
    }

    @Test
    internal fun `generalises to an unseen record sharing a discriminating feature`() {
        val model = learner().fit(observations = toyCorpus())

        // Feature 3 never occurred in training; feature 0 is ALPHA's marker.
        val prediction = model.predict(features = features(0, 3), threshold = 0.5)

        assertEquals(expected = setOf(alpha), actual = prediction.labels())
    }

    @Test
    internal fun `orders labels deterministically regardless of corpus order`() {
        val forward = learner().fit(observations = toyCorpus())
        val reversed = learner().fit(observations = toyCorpus().reversed())

        val expected = listOf(alpha, beta)
        assertEquals(expected = expected, actual = (forward as MultiLabelLogisticClassifier).weights().labels)
        assertEquals(expected = expected, actual = (reversed as MultiLabelLogisticClassifier).weights().labels)
    }

    @Test
    internal fun `produces identical scores across repeated fits of the same corpus`() {
        val first = learner().fit(observations = toyCorpus())
        val second = learner().fit(observations = toyCorpus())

        val probe = features(0, 1)
        assertEquals(expected = first.logits(features = probe), actual = second.logits(features = probe))
    }

    @Test
    internal fun `reports exactly the labels present in the corpus`() {
        val model = learner().fit(observations = toyCorpus())

        assertEquals(expected = setOf(alpha, beta), actual = model.labels())
    }

    @Test
    internal fun `pruning shrinks the stored matrix`() {
        val dense = learner(keepFraction = 1.0).fit(observations = toyCorpus())
        val sparse = learner(keepFraction = 0.25).fit(observations = toyCorpus())

        val denseCount = (dense as MultiLabelLogisticClassifier).weights().nonZeroCount()
        val sparseCount = (sparse as MultiLabelLogisticClassifier).weights().nonZeroCount()

        assertTrue(actual = sparseCount < denseCount, message = "$sparseCount was not below $denseCount")
    }

    /** The decomposition is exact for a one-vs-rest head, unlike the centered single-label one. */
    @Test
    internal fun `explains a prediction as an exact sum of intercept and contributions`() {
        val model = learner().fit(observations = toyCorpus())
        val probe = features(0, 4)

        val explanation = model.explain(features = probe, label = alpha, limit = 10)!!

        assertEquals(
            expected = model.logits(features = probe).getValue(key = alpha),
            actual = explanation.total,
            absoluteTolerance = 1e-9,
        )
        assertEquals(
            expected = explanation.total,
            actual = explanation.base + explanation.contributions.sumOf { entry -> entry.contribution },
            absoluteTolerance = 1e-9,
        )
    }

    @Test
    internal fun `explanation contributions are ranked by absolute magnitude`() {
        val model = learner().fit(observations = toyCorpus())

        val explanation = model.explain(features = features(0, 1, 4), label = alpha, limit = 10)!!

        val magnitudes = explanation.contributions.map { entry -> abs(x = entry.contribution) }
        assertEquals(expected = magnitudes.sortedDescending(), actual = magnitudes)
    }

    @Test
    internal fun `cannot explain a label the model does not know`() {
        val model = learner().fit(observations = toyCorpus())

        assertNull(actual = model.explain(features = features(0), label = Label(value = "UNKNOWN"), limit = 5))
    }

    @Test
    internal fun `records the regularisation it was fitted with`() {
        val model = learner().fit(observations = toyCorpus())

        assertEquals(
            expected = 0.1,
            actual = model.hyperparameters().l2Regularization,
            absoluteTolerance = 1e-12,
        )
    }

    @Test
    internal fun `rejects an empty corpus`() {
        assertFailsWith<IllegalArgumentException> { learner().fit(observations = emptyList()) }
    }

    @Test
    internal fun `rejects a corpus in which no observation carries a label`() {
        assertFailsWith<IllegalArgumentException> {
            learner().fit(observations = listOf(observation(0, labels = emptySet())))
        }
    }

    @Test
    internal fun `rejects an invalid configuration`() {
        assertFailsWith<IllegalArgumentException> { LbfgsMultiLabelLearner(featureCount = 0) }
        assertFailsWith<IllegalArgumentException> {
            LbfgsMultiLabelLearner(featureCount = 4, inverseRegularization = 0.0)
        }
        assertFailsWith<IllegalArgumentException> {
            LbfgsMultiLabelLearner(featureCount = 4, keepFraction = 0.0)
        }
    }
}
