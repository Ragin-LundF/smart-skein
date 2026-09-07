package io.skein.classify.application

import io.skein.classify.domain.Calibration
import io.skein.classify.domain.ClassifierHyperparameters
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import io.skein.classify.infrastructure.NaiveBayesClassifier
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.deleteExisting
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

internal class ModelStoreHyperparametersTest {

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }
    private val hashing = HashingConfig(key0 = 5L, key1 = 6L)

    private val tuned = ClassifierHyperparameters(
        smoothingAlpha = 0.35,
        initialLearningRate = 0.9,
        decayRate = 0.5,
        l2Regularization = 0.3,
    )

    private val observations = List(size = 20) { index ->
        LabeledFeatures(
            label = Label(value = if (index % 2 == 0) "A" else "B"),
            features = FeatureVector(
                indices = intArrayOf(index % 2 + 1),
                values = floatArrayOf(1.0f),
            ),
        )
    }

    private val probe = FeatureVector(indices = intArrayOf(1), values = floatArrayOf(1.0f))

    private fun saveTuned(): Path {
        val path = Files.createTempFile("skein-hyper", ".skein")
        ModelStore.save(
            path = path,
            schema = schema,
            classifier = ClassifierKindEnum.LOGISTIC_REGRESSION,
            hashingConfig = hashing,
            observations = observations,
            calibration = Calibration.NONE,
            hyperparameters = tuned,
        )
        return path
    }

    @Test
    internal fun `hyperparameters round-trip through the model file`() {
        val path = saveTuned()
        try {
            assertEquals(expected = tuned, actual = ModelStore.load(path = path).hyperparameters)
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `a restored tuned model predicts like the original`() {
        // The regression this guards: loading replays the stored observations through a freshly
        // built classifier, so losing the tuning silently produces a different model.
        val original = LogisticRegressionSgdClassifier(
            initialLearningRate = tuned.initialLearningRate,
            decayRate = tuned.decayRate,
            l2Regularization = tuned.l2Regularization,
        )
        original.learnAll(observations = observations)

        val path = saveTuned()
        try {
            val loaded = ModelStore.load(path = path)
            val restored = ClassifierFactory.create(
                kind = loaded.classifier,
                hyperparameters = loaded.hyperparameters,
            )
            restored.learnAll(observations = loaded.observations)

            assertEquals(
                expected = original.classify(features = probe).confidence,
                actual = restored.classify(features = probe).confidence,
                absoluteTolerance = 1e-12,
            )
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `a model saved without hyperparameters loads the defaults`() {
        val path = Files.createTempFile("skein-plain-hyper", ".skein")
        try {
            ModelStore.save(
                path = path,
                schema = schema,
                classifier = ClassifierKindEnum.NAIVE_BAYES,
                hashingConfig = hashing,
                observations = observations,
            )
            assertEquals(
                expected = ClassifierHyperparameters.DEFAULTS,
                actual = ModelStore.load(path = path).hyperparameters,
            )
        } finally {
            path.deleteExisting()
        }
    }

    @Test
    internal fun `classifiers report the tuning they were built with`() {
        assertEquals(
            expected = 0.35,
            actual = NaiveBayesClassifier(smoothingAlpha = 0.35).hyperparameters().smoothingAlpha,
        )
        val logistic = LogisticRegressionSgdClassifier(
            initialLearningRate = 0.9,
            decayRate = 0.5,
            l2Regularization = 0.3,
        ).hyperparameters()
        assertEquals(expected = 0.9, actual = logistic.initialLearningRate)
        assertEquals(expected = 0.5, actual = logistic.decayRate)
        assertEquals(expected = 0.3, actual = logistic.l2Regularization)
    }

    @Test
    internal fun `an untuned classifier reports the library defaults`() {
        assertEquals(expected = ClassifierHyperparameters.DEFAULTS, actual = NaiveBayesClassifier().hyperparameters())
        assertEquals(
            expected = ClassifierHyperparameters.DEFAULTS,
            actual = LogisticRegressionSgdClassifier().hyperparameters(),
        )
    }

    @Test
    internal fun `rejects nonsensical hyperparameters`() {
        assertFailsWith<IllegalArgumentException> { ClassifierHyperparameters(smoothingAlpha = 0.0) }
        assertFailsWith<IllegalArgumentException> { ClassifierHyperparameters(initialLearningRate = 0.0) }
        assertFailsWith<IllegalArgumentException> { ClassifierHyperparameters(decayRate = -1.0) }
        assertFailsWith<IllegalArgumentException> { ClassifierHyperparameters(l2Regularization = -1.0) }
    }
}
