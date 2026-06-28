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

internal class ClassificationServiceTest {

    private val schema = Schema.define {
        text(name = "purpose")
        identifier(name = "iban")
        label(name = "category")
    }

    private fun record(purpose: String, category: String? = null): Record {
        val values = HashMap<String, Any?>()
        values["purpose"] = purpose
        values["iban"] = "DE00000000"
        if (category != null) {
            values["category"] = category
        }
        return Record(values = values)
    }

    private fun trainedService(): ClassificationService {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        service.learnAll(
            records = listOf(
                record(purpose = "rent transfer landlord", category = "housing"),
                record(purpose = "monthly apartment rent", category = "housing"),
                record(purpose = "salary october payout", category = "income"),
                record(purpose = "monthly salary payment", category = "income"),
            ),
        )
        return service
    }

    @Test
    internal fun `learns then classifies an unseen record`() {
        val prediction = trainedService().classify(record = record(purpose = "rent payment for apartment"))
        assertEquals(expected = Label(value = "housing"), actual = prediction.label)
    }

    @Test
    internal fun `learning rejects a record without a label`() {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        assertFailsWith<IllegalArgumentException> {
            service.learn(record = record(purpose = "no label here"))
        }
    }

    @Test
    internal fun `metrics reports total and per-label counts`() {
        val metrics = trainedService().metrics()
        assertEquals(expected = 4, actual = metrics.totalObservations)
        assertEquals(expected = 2, actual = metrics.perLabelCounts[Label(value = "housing")])
        assertEquals(expected = 2, actual = metrics.perLabelCounts[Label(value = "income")])
    }

    @Test
    internal fun `feedback teaches a corrected label`() {
        val service = trainedService()
        repeat(times = 3) {
            service.feedback(
                record = record(purpose = "salary october payout"),
                correctLabel = Label(value = "housing"),
            )
        }
        // After repeated correction the previously-income text now leans housing.
        assertEquals(
            expected = Label(value = "housing"),
            actual = service.classify(record = record(purpose = "salary october payout")).label,
        )
    }

    @Test
    internal fun `forget resets the engine`() {
        val service = trainedService()
        service.forget()
        assertEquals(expected = 0, actual = service.metrics().totalObservations)
    }

    @Test
    internal fun `learnAll rejects a record without a label value`() {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        assertFailsWith<IllegalArgumentException> {
            service.learnAll(records = listOf(record(purpose = "rent transfer")))
        }
    }
}
