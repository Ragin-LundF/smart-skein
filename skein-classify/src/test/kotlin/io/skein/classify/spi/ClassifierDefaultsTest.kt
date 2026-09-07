package io.skein.classify.spi

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Prediction
import io.skein.classify.domain.PredictionFactory
import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * A third-party [Classifier] implementing only the members that existed before `logScores` and
 * `explain` were added. That this class compiles at all is the source-compatibility proof; the
 * assertions cover the behaviour the interface defaults are supposed to provide.
 */
private class MinimalClassifier : Classifier {

    private val counts = HashMap<Label, Int>()

    override fun learn(features: FeatureVector, label: Label) {
        counts[label] = (counts[label] ?: 0) + 1
    }

    override fun classify(features: FeatureVector): Prediction {
        val total = counts.values.sum().toDouble()
        return PredictionFactory.fromLogScores(
            logScores = counts.mapValues { (_, count) -> ln(x = count / total) },
        )
    }

    override fun labels(): Set<Label> {
        return counts.keys
    }

    override fun forget() {
        counts.clear()
    }
}

internal class ClassifierDefaultsTest {

    private val probe = FeatureVector(indices = intArrayOf(1), values = floatArrayOf(1.0f))

    private fun trained(): Classifier {
        val classifier = MinimalClassifier()
        classifier.learnAll(
            observations = List(size = 3) { LabeledFeatures(label = Label(value = "A"), features = probe) } +
                List(size = 1) { LabeledFeatures(label = Label(value = "B"), features = probe) },
        )
        return classifier
    }

    @Test
    internal fun `the default logScores agrees with classify on ranking and confidence`() {
        val classifier = trained()
        val prediction = classifier.classify(features = probe)
        val derived = PredictionFactory.fromLogScores(logScores = classifier.logScores(features = probe))

        assertEquals(expected = prediction.label, actual = derived.label)
        assertEquals(expected = prediction.confidence, actual = derived.confidence, absoluteTolerance = 1e-9)
    }

    @Test
    internal fun `the default logScores recovers the probabilities`() {
        val classifier = trained()
        val scores = classifier.logScores(features = probe)

        classifier.classify(features = probe).alternatives.forEach { scored ->
            assertEquals(
                expected = scored.probability,
                actual = exp(x = scores.getValue(key = scored.label)),
                absoluteTolerance = 1e-9,
            )
        }
    }

    @Test
    internal fun `the default explain declines rather than throwing`() {
        assertNull(actual = trained().explain(features = probe, label = Label(value = "A"), limit = 5))
    }

    @Test
    internal fun `the default learnAll loops over learn`() {
        val classifier = trained()
        assertTrue(actual = Label(value = "A") in classifier.labels())
        assertTrue(actual = Label(value = "B") in classifier.labels())
    }
}
