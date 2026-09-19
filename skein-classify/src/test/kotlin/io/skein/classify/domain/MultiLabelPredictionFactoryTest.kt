package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class MultiLabelPredictionFactoryTest {

    private val tolerance = 1e-12

    @Test
    internal fun `applies an independent sigmoid per label rather than a softmax`() {
        val prediction = MultiLabelPredictionFactory.fromLogits(
            logits = mapOf(Label(value = "A") to 0.0, Label(value = "B") to 0.0, Label(value = "C") to 0.0),
            threshold = 0.5,
        )

        // A softmax over three equal logits would give 1/3 each and sum to 1. Independent sigmoids
        // give 0.5 each and sum to 1.5, which is the whole point of the port.
        prediction.ranked.forEach { scored ->
            assertEquals(expected = 0.5, actual = scored.probability, absoluteTolerance = tolerance)
        }
        assertEquals(
            expected = 1.5,
            actual = prediction.ranked.sumOf { scored -> scored.probability },
            absoluteTolerance = tolerance,
        )
    }

    @Test
    internal fun `ranks by probability descending and breaks ties by label value`() {
        val prediction = MultiLabelPredictionFactory.fromLogits(
            logits = mapOf(
                Label(value = "zebra") to 1.0,
                Label(value = "alpha") to 1.0,
                Label(value = "winner") to 3.0,
            ),
            threshold = 0.5,
        )

        assertEquals(
            expected = listOf("winner", "alpha", "zebra"),
            actual = prediction.ranked.map { scored -> scored.label.value },
        )
    }

    @Test
    internal fun `stays finite at margins that overflow a naive exponential`() {
        // exp(800) is Infinity in double precision, so 1 / (1 + exp(-z)) returns NaN at z = -800.
        assertEquals(
            expected = 1.0,
            actual = MultiLabelPredictionFactory.logistic(logit = 800.0),
            absoluteTolerance = tolerance,
        )
        assertEquals(
            expected = 0.0,
            actual = MultiLabelPredictionFactory.logistic(logit = -800.0),
            absoluteTolerance = tolerance,
        )
        assertTrue(actual = MultiLabelPredictionFactory.logistic(logit = -800.0).isFinite())
        assertTrue(actual = MultiLabelPredictionFactory.logistic(logit = 800.0).isFinite())
    }

    @Test
    internal fun `is symmetric about zero`() {
        listOf(0.25, 1.0, 5.0, 40.0).forEach { logit ->
            assertEquals(
                expected = 1.0,
                actual = MultiLabelPredictionFactory.logistic(logit = logit) +
                    MultiLabelPredictionFactory.logistic(logit = -logit),
                absoluteTolerance = 1e-15,
            )
        }
    }

    @Test
    internal fun `produces an empty prediction for a model that knows no labels`() {
        val prediction = MultiLabelPredictionFactory.fromLogits(logits = emptyMap(), threshold = 0.5)

        assertTrue(actual = prediction.ranked.isEmpty())
        assertTrue(actual = prediction.labels().isEmpty())
    }
}
