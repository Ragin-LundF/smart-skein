package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.infrastructure.LogisticRegressionSgdClassifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ClassificationServiceRetrainTest {

    private val schema = Schema.define {
        text("purpose")
        label("category")
    }

    private fun service(): ClassificationService {
        return ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
            classifier = LogisticRegressionSgdClassifier(),
        )
    }

    @Test
    fun `retrain over multiple epochs converges the logistic model from the store`() {
        val service = service()
        service.learnAll(
            listOf(
                Record(mapOf("purpose" to "rent transfer landlord", "category" to "housing")),
                Record(mapOf("purpose" to "monthly apartment rent", "category" to "housing")),
                Record(mapOf("purpose" to "salary october payout", "category" to "income")),
                Record(mapOf("purpose" to "monthly salary payment", "category" to "income")),
            ),
        )

        service.retrain(epochs = 80)

        val prediction = service.classify(Record(mapOf("purpose" to "apartment rent payment")))
        assertEquals(expected = Label("housing"), actual = prediction.label)
        // Retrain keeps the stored observations intact.
        assertEquals(expected = 4, actual = service.metrics().totalObservations)
    }

    @Test
    fun `retrain rejects a non-positive epoch count`() {
        assertFailsWith<IllegalArgumentException> {
            service().retrain(epochs = 0)
        }
    }

    @Test
    fun `seeded shuffle retrain is deterministic and still classifies`() {
        val records = listOf(
            Record(mapOf("purpose" to "rent transfer landlord", "category" to "housing")),
            Record(mapOf("purpose" to "monthly apartment rent", "category" to "housing")),
            Record(mapOf("purpose" to "salary october payout", "category" to "income")),
            Record(mapOf("purpose" to "monthly salary payment", "category" to "income")),
        )
        val sample = Record(mapOf("purpose" to "apartment rent payment"))

        val first = service().also { it.learnAll(records); it.retrain(epochs = 80, seed = 42L) }.classify(sample)
        val second = service().also { it.learnAll(records); it.retrain(epochs = 80, seed = 42L) }.classify(sample)

        assertEquals(expected = Label("housing"), actual = first.label)
        // Same seed → identical confidence (deterministic shuffle).
        assertEquals(expected = first.confidence, actual = second.confidence, absoluteTolerance = 1e-12)
    }
}
