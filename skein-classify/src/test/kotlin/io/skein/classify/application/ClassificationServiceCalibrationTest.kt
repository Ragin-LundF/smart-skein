package io.skein.classify.application

import io.skein.classify.domain.Calibration
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

internal class ClassificationServiceCalibrationTest {

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }

    private fun trainedService(): ClassificationService {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 3L, key1 = 5L),
        )
        listOf(
            "rent transfer landlord" to "RENT",
            "monthly rent landlord" to "RENT",
            "supermarket groceries" to "FOOD",
            "groceries market food" to "FOOD",
        ).forEach { (purpose, category) ->
            service.learn(record = Record(values = mapOf("purpose" to purpose, "category" to category)))
        }
        return service
    }

    private val probe = Record(values = mapOf("purpose" to "rent transfer landlord"))

    @Test
    internal fun `defaults to no calibration`() {
        assertEquals(expected = Calibration.NONE, actual = trainedService().calibration)
    }

    @Test
    internal fun `calibration changes confidence but not the predicted label`() {
        val service = trainedService()
        val before = service.classify(record = probe)

        service.calibration = Calibration(temperature = 25.0)
        val after = service.classify(record = probe)

        assertEquals(expected = before.label, actual = after.label)
        assertTrue(actual = after.confidence < before.confidence)
    }

    @Test
    internal fun `classifyOrNull abstains below the threshold and returns above it`() {
        val service = trainedService()
        service.calibration = Calibration(temperature = 50.0)
        val confidence = service.classify(record = probe).confidence

        assertNull(actual = service.classifyOrNull(record = probe, minConfidence = confidence + 0.01))
        assertNotNull(actual = service.classifyOrNull(record = probe, minConfidence = confidence))
        assertNotNull(actual = service.classifyOrNull(record = probe, minConfidence = 0.0))
    }

    @Test
    internal fun `fitCalibration installs and returns the fitted temperature`() {
        val service = trainedService()
        val heldOut = service.featureStore.all()

        val fitted = service.fitCalibration(heldOut = heldOut)

        assertEquals(expected = fitted, actual = service.calibration)
    }
}
