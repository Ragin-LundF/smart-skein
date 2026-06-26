package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ClassificationServiceTest {

    private val schema = Schema.define {
        text("purpose")
        identifier("iban")
        label("category")
    }

    private fun record(purpose: String, category: String? = null): Record {
        val values = HashMap<String, Any?>()
        values["purpose"] = purpose
        values["iban"] = "DE00000000"
        if (category != null) {
            values["category"] = category
        }
        return Record(values)
    }

    private fun trainedService(): ClassificationService {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        service.learnAll(
            listOf(
                record("rent transfer landlord", "housing"),
                record("monthly apartment rent", "housing"),
                record("salary october payout", "income"),
                record("monthly salary payment", "income"),
            ),
        )
        return service
    }

    @Test
    fun `learns then classifies an unseen record`() {
        val prediction = trainedService().classify(record("rent payment for apartment"))
        assertEquals(expected = Label("housing"), actual = prediction.label)
    }

    @Test
    fun `learning rejects a record without a label`() {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        assertFailsWith<IllegalArgumentException> {
            service.learn(record("no label here"))
        }
    }

    @Test
    fun `metrics reports total and per-label counts`() {
        val metrics = trainedService().metrics()
        assertEquals(expected = 4, actual = metrics.totalObservations)
        assertEquals(expected = 2, actual = metrics.perLabelCounts[Label("housing")])
        assertEquals(expected = 2, actual = metrics.perLabelCounts[Label("income")])
    }

    @Test
    fun `feedback teaches a corrected label`() {
        val service = trainedService()
        repeat(3) { service.feedback(record("salary october payout"), Label("housing")) }
        // After repeated correction the previously-income text now leans housing.
        assertEquals(expected = Label("housing"), actual = service.classify(record("salary october payout")).label)
    }

    @Test
    fun `forget resets the engine`() {
        val service = trainedService()
        service.forget()
        assertEquals(expected = 0, actual = service.metrics().totalObservations)
    }
}
