package io.skein.cli

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.io.path.createTempFile
import kotlin.io.path.deleteIfExists
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals

class ModelStoreTest {

    private val modelFile = createTempFile(prefix = "skein-model", suffix = ".txt")

    private val schema = Schema.define {
        text(name = "purpose")
        identifier(name = "iban")
        label(name = "category")
    }
    private val hashing = HashingConfig(key0 = 11L, key1 = 22L)
    private val holdout = Record(values = mapOf("purpose" to "insurance premium policy", "iban" to "DE00"))

    @AfterTest
    fun cleanup() {
        modelFile.deleteIfExists()
    }

    @Test
    fun `save then load reproduces predictions (Naive Bayes)`() {
        assertRoundTrips(classifier = ClassifierKindEnum.NAIVE_BAYES)
    }

    @Test
    fun `save then load reproduces predictions (logistic regression)`() {
        assertRoundTrips(classifier = ClassifierKindEnum.LOGISTIC_REGRESSION)
    }

    private fun assertRoundTrips(classifier: ClassifierKindEnum) {
        val original = CliEngine.fresh(schema = schema, classifier = classifier, hashingConfig = hashing)
        train(engine = original)
        original.service.retrain(epochs = 5)
        val before = original.service.classify(record = holdout)

        original.save(path = modelFile)
        val restored = CliEngine.restore(model = ModelStore.load(path = modelFile), epochs = 5)
        val after = restored.service.classify(record = holdout)

        assertEquals(expected = before.label, actual = after.label)
        assertEquals(expected = before.confidence, actual = after.confidence, absoluteTolerance = 1e-9)
        assertEquals(
            expected = original.service.metrics().totalObservations,
            actual = restored.service.metrics().totalObservations,
        )
    }

    private fun train(engine: CliEngine) {
        engine.service.learnAll(
            records = listOf(
                sample(purpose = "insurance premium annual policy", iban = "DE01", category = "insurance"),
                sample(purpose = "AIG insurance premium", iban = "DE02", category = "insurance"),
                sample(purpose = "rent apartment payment", iban = "DE03", category = "rent"),
                sample(purpose = "monthly rent apartment", iban = "DE04", category = "rent"),
            ),
        )
    }

    private fun sample(purpose: String, iban: String, category: String): Record {
        return Record(values = mapOf("purpose" to purpose, "iban" to iban, "category" to category))
    }
}
