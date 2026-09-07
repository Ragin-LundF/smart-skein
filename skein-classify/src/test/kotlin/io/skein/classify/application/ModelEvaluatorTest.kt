package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.NaiveBayesClassifier
import io.skein.classify.spi.Classifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class ModelEvaluatorTest {

    private val tolerance = 1e-9

    /** Two linearly separable classes: label A fires feature 1, label B fires feature 2. */
    private fun separableCorpus(perLabel: Int): List<LabeledFeatures> {
        return List(size = perLabel) {
            LabeledFeatures(
                label = Label(value = "A"),
                features = FeatureVector(indices = intArrayOf(1), values = floatArrayOf(1.0f)),
            )
        } + List(size = perLabel) {
            LabeledFeatures(
                label = Label(value = "B"),
                features = FeatureVector(indices = intArrayOf(2), values = floatArrayOf(1.0f)),
            )
        }
    }

    private fun trainedOn(observations: List<LabeledFeatures>): Classifier {
        val classifier = NaiveBayesClassifier()
        classifier.learnAll(observations = observations)
        return classifier
    }

    @Test
    internal fun `evaluate scores a trained classifier on separable data`() {
        val corpus = separableCorpus(perLabel = 10)
        val report = ModelEvaluator().evaluate(classifier = trainedOn(observations = corpus), holdout = corpus)

        assertEquals(expected = corpus.size, actual = report.sampleCount)
        assertEquals(expected = 1.0, actual = report.accuracy, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `evaluate does not train the classifier it scores`() {
        val classifier = trainedOn(observations = separableCorpus(perLabel = 5))
        val before = classifier.labels()

        ModelEvaluator().evaluate(
            classifier = classifier,
            holdout = listOf(
                LabeledFeatures(
                    label = Label(value = "unseen"),
                    features = FeatureVector(indices = intArrayOf(3), values = floatArrayOf(1.0f)),
                ),
            ),
        )

        assertEquals(expected = before, actual = classifier.labels())
        assertTrue(actual = Label(value = "unseen") !in classifier.labels())
    }

    @Test
    internal fun `holdout builds exactly one fresh classifier`() {
        var built = 0
        ModelEvaluator().holdout(
            observations = separableCorpus(perLabel = 20),
            classifierFactory = {
                built += 1
                NaiveBayesClassifier()
            },
        )
        assertEquals(expected = 1, actual = built)
    }

    @Test
    internal fun `cross-validation builds one fresh classifier per fold`() {
        var built = 0
        val report = ModelEvaluator().crossValidate(
            observations = separableCorpus(perLabel = 20),
            classifierFactory = {
                built += 1
                NaiveBayesClassifier()
            },
            folds = 5,
        )

        assertEquals(expected = 5, actual = built)
        assertEquals(expected = 5, actual = report.folds.size)
    }

    @Test
    internal fun `cross-validation holds out every observation exactly once`() {
        val corpus = separableCorpus(perLabel = 20)
        val report = ModelEvaluator().crossValidate(
            observations = corpus,
            classifierFactory = { NaiveBayesClassifier() },
            folds = 5,
        )

        assertEquals(expected = corpus.size, actual = report.pooled.sampleCount)
        assertEquals(expected = corpus.size, actual = report.folds.sumOf { fold -> fold.sampleCount })
    }

    @Test
    internal fun `cross-validation is deterministic for a fixed seed`() {
        val corpus = separableCorpus(perLabel = 15)
        val first = ModelEvaluator().crossValidate(
            observations = corpus,
            classifierFactory = { NaiveBayesClassifier() },
        )
        val second = ModelEvaluator().crossValidate(
            observations = corpus,
            classifierFactory = { NaiveBayesClassifier() },
        )

        assertEquals(
            expected = first.folds.map { fold -> fold.accuracy },
            actual = second.folds.map { fold -> fold.accuracy },
        )
    }

    @Test
    internal fun `evaluateRecords agrees with evaluate on the same data`() {
        val schema = Schema.define {
            text(name = "purpose")
            label(name = "category")
        }
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        val records = listOf(
            Record(values = mapOf("purpose" to "rent transfer landlord", "category" to "RENT")),
            Record(values = mapOf("purpose" to "supermarket groceries", "category" to "FOOD")),
        )
        records.forEach { record -> service.learn(record = record) }

        val report = ModelEvaluator().evaluateRecords(service = service, records = records)

        assertEquals(expected = 2, actual = report.sampleCount)
        assertEquals(expected = 1.0, actual = report.accuracy, absoluteTolerance = tolerance)
    }

    @Test
    internal fun `evaluateRecords rejects a record without a label`() {
        val schema = Schema.define {
            text(name = "purpose")
            label(name = "category")
        }
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        service.learn(record = Record(values = mapOf("purpose" to "rent", "category" to "RENT")))

        val failure = assertFailsWith<IllegalArgumentException> {
            ModelEvaluator().evaluateRecords(
                service = service,
                records = listOf(Record(values = mapOf("purpose" to "rent"))),
            )
        }
        assertTrue(actual = failure.message.orEmpty().contains(other = "category"))
    }

    @Test
    internal fun `rejects empty inputs`() {
        assertFailsWith<IllegalArgumentException> {
            ModelEvaluator().evaluate(classifier = NaiveBayesClassifier(), holdout = emptyList())
        }
        assertFailsWith<IllegalArgumentException> {
            ModelEvaluator().crossValidate(
                observations = separableCorpus(perLabel = 4),
                classifierFactory = { NaiveBayesClassifier() },
                folds = 1,
            )
        }
    }
}
