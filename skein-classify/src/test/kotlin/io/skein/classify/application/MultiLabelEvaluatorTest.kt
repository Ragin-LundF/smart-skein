package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.spi.MultiLabelClassifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class MultiLabelEvaluatorTest {

    private val tolerance = 1e-9
    private val evaluator = MultiLabelEvaluator()

    /**
     * Scores a record by the single feature index it carries, so a fixture can dictate exactly what
     * the model returns without any fitting. Index 0 means "A is certain", index 1 means "B is".
     */
    private class ScriptedClassifier(
        private val logitsByIndex: Map<Int, Map<Label, Double>>,
    ) : MultiLabelClassifier {

        override fun logits(features: FeatureVector): Map<Label, Double> {
            return logitsByIndex.getValue(key = features.indices.first())
        }

        override fun labels(): Set<Label> {
            return logitsByIndex.values.flatMap { scores -> scores.keys }.toSet()
        }
    }

    private fun observation(index: Int, vararg labels: String): MultiLabeledFeatures {
        return MultiLabeledFeatures(
            features = FeatureVector(indices = intArrayOf(index), values = floatArrayOf(1.0f)),
            labels = labels.map { value -> Label(value = value) }.toSet(),
        )
    }

    private val a = Label(value = "A")
    private val b = Label(value = "B")

    /**
     * Record 0 is strongly A and weakly B; record 1 is the reverse. Truth says record 0 is both.
     * That makes B's score the one the threshold decides, which is what the sweep should show.
     */
    private fun scriptedClassifier(): MultiLabelClassifier {
        return ScriptedClassifier(
            logitsByIndex = mapOf(
                0 to mapOf(a to 4.0, b to 0.4),
                1 to mapOf(a to -4.0, b to 4.0),
            ),
        )
    }

    private fun holdout(): List<MultiLabeledFeatures> {
        return listOf(observation(index = 0, "A", "B"), observation(index = 1, "B"))
    }

    @Test
    internal fun `evaluate scores the model it is handed at the given threshold`() {
        val metrics = evaluator.evaluate(
            classifier = scriptedClassifier(),
            holdout = holdout(),
            threshold = 0.5,
        )

        // sigmoid(0.4) = 0.599 clears 0.5, so record 0 gets both labels and record 1 gets B.
        assertEquals(expected = 2, actual = metrics.sampleCount)
        assertEquals(expected = 1.0, actual = metrics.micro.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 1.0, actual = metrics.micro.recall, absoluteTolerance = tolerance)
        assertEquals(expected = 1.0, actual = metrics.exactMatchRatio, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `raising the threshold trades recall for precision and loses a label`() {
        val metrics = evaluator.evaluate(
            classifier = scriptedClassifier(),
            holdout = holdout(),
            threshold = 0.7,
        )

        // sigmoid(0.4) = 0.599 now falls short, so record 0 loses B: 2 of 3 expected labels found.
        assertEquals(expected = 1.0, actual = metrics.micro.precision, absoluteTolerance = tolerance)
        assertEquals(expected = 2.0 / 3.0, actual = metrics.micro.recall, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `sweep re-reads one scoring pass at every threshold in ascending order`() {
        val outcomes = evaluator.outcomes(
            classifier = scriptedClassifier(),
            holdout = holdout(),
            threshold = 0.5,
        )

        val points = evaluator.sweep(outcomes = outcomes, thresholds = listOf(0.7, 0.3))

        assertEquals(expected = listOf(0.3, 0.7), actual = points.map { point -> point.threshold })
        assertEquals(expected = 1.0, actual = points.first().recall, absoluteTolerance = tolerance)
        assertEquals(expected = 2.0 / 3.0, actual = points.last().recall, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `sweep reports falling coverage as the threshold rises`() {
        val outcomes = evaluator.outcomes(
            classifier = scriptedClassifier(),
            holdout = holdout(),
            threshold = 0.5,
        )

        val points = evaluator.sweep(outcomes = outcomes, thresholds = listOf(0.3, 0.999))

        assertEquals(expected = 1.0, actual = points.first().coverage, absoluteTolerance = tolerance)
        assertEquals(expected = 0.0, actual = points.last().coverage, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `sweep uses a default threshold ladder when none is given`() {
        val outcomes = evaluator.outcomes(
            classifier = scriptedClassifier(),
            holdout = holdout(),
            threshold = 0.5,
        )

        val points = evaluator.sweep(outcomes = outcomes)

        assertEquals(expected = listOf(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), actual = points.map { it.threshold })
        assertTrue(actual = points.zipWithNext().all { (low, high) -> low.recall >= high.recall })
    }

    @Test
    internal fun `recall at k is monotonically non-decreasing in depth`() {
        val outcomes = evaluator.outcomes(
            classifier = scriptedClassifier(),
            holdout = holdout(),
            threshold = 0.5,
        )

        val recalls = evaluator.recallAtK(outcomes = outcomes, depths = listOf(2, 1))

        assertEquals(expected = listOf(1, 2), actual = recalls.keys.toList())
        assertEquals(expected = 2.0 / 3.0, actual = recalls.getValue(key = 1), absoluteTolerance = tolerance)
        assertEquals(expected = 1.0, actual = recalls.getValue(key = 2), absoluteTolerance = tolerance)
    }

    @Test
    internal fun `rejects an empty holdout`() {
        assertFailsWith<IllegalArgumentException> {
            evaluator.evaluate(classifier = scriptedClassifier(), holdout = emptyList(), threshold = 0.5)
        }
    }

    @Test
    internal fun `rejects an empty sweep`() {
        val outcomes = evaluator.outcomes(
            classifier = scriptedClassifier(),
            holdout = holdout(),
            threshold = 0.5,
        )

        assertFailsWith<IllegalArgumentException> { evaluator.sweep(outcomes = emptyList()) }
        assertFailsWith<IllegalArgumentException> {
            evaluator.sweep(outcomes = outcomes, thresholds = emptyList())
        }
        assertFailsWith<IllegalArgumentException> { evaluator.recallAtK(outcomes = emptyList()) }
    }

    @Test
    internal fun `scoreAll derives independent probabilities from the logits an implementation returns`() {
        val classifier = scriptedClassifier()
        val features = FeatureVector(indices = intArrayOf(0), values = floatArrayOf(1.0f))

        val probabilities = classifier.scoreAll(features = features)

        assertEquals(expected = setOf(a, b), actual = probabilities.keys)
        assertTrue(actual = probabilities.values.sumOf { probability -> probability } > 1.0)
    }
}
