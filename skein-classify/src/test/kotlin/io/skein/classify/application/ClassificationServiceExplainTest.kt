package io.skein.classify.application

import io.skein.classify.domain.AttributionModeEnum
import io.skein.classify.domain.Calibration
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

internal class ClassificationServiceExplainTest {

    private val schema = Schema.define {
        text(name = "purpose")
        identifier(name = "iban")
        label(name = "category")
    }

    private fun trainedService(): ClassificationService {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 21L, key1 = 34L),
        )
        listOf(
            "rent transfer landlord" to "RENT",
            "monthly rent landlord apartment" to "RENT",
            "supermarket groceries food" to "FOOD",
            "groceries market shopping" to "FOOD",
        ).forEach { (purpose, category) ->
            service.learn(
                record = Record(values = mapOf("purpose" to purpose, "iban" to "DE00", "category" to category)),
            )
        }
        return service
    }

    private val probe = Record(values = mapOf("purpose" to "rent transfer landlord", "iban" to "DE99"))

    @Test
    internal fun `the explained probability equals the calibrated confidence`() {
        val service = trainedService()
        service.calibration = Calibration(temperature = 7.5)

        val prediction = service.classify(record = probe)
        val explanation = service.explain(record = probe)!!

        assertEquals(expected = prediction.label, actual = explanation.label)
        assertEquals(
            expected = prediction.confidence,
            actual = explanation.probability,
            absoluteTolerance = 1e-12,
        )
    }

    @Test
    internal fun `buckets-only is the default and reveals no text`() {
        val explanation = trainedService().explain(record = probe)!!

        assertTrue(actual = explanation.contributions.isNotEmpty())
        explanation.contributions.forEach { contribution -> assertNull(actual = contribution.ngram) }
    }

    @Test
    internal fun `with-ngrams resolves buckets to fragments of the record`() {
        val service = trainedService()
        val explanation = service.explain(record = probe, mode = AttributionModeEnum.WITH_NGRAMS)!!

        val resolved = explanation.contributions.mapNotNull { contribution -> contribution.ngram }
        assertTrue(actual = resolved.isNotEmpty())

        // Every resolved fragment must come from the record's own feature text.
        val featureText = "rent transfer landlord"
        resolved.forEach { ngram ->
            assertTrue(actual = featureText.contains(other = ngram), message = "unexpected fragment '$ngram'")
        }
    }

    @Test
    internal fun `attribution never exposes a PII field`() {
        val service = trainedService()
        val explanation = service.explain(
            record = probe,
            limit = Int.MAX_VALUE,
            mode = AttributionModeEnum.WITH_NGRAMS,
        )!!

        // The iban field is PII, so RecordMapper excludes it from the feature text entirely.
        explanation.contributions.mapNotNull { it.ngram }.forEach { ngram ->
            assertTrue(actual = !ngram.contains(other = "DE99"), message = "PII leaked via '$ngram'")
        }
    }

    @Test
    internal fun `honours the contribution limit`() {
        val explanation = trainedService().explain(record = probe, limit = 3)!!
        assertEquals(expected = 3, actual = explanation.contributions.size)
    }
}
